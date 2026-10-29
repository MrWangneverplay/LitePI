// litepi_seal.cpp - pybind11 CKKS layer for LitePI on top of SEAL.
//
// SEAL 3.3.2 and 4.1.x are both supported: build.sh links the 3.3.2 static
// library shipped with SCI, build4.sh links SEAL 4.1.2 (the version named in the
// paper's implementation section).  The API differences are confined to the
// context/key setup and the parameter serialization below.
//
// The LitePI protocol needs a narrow CKKS interface: encrypt/decrypt, plaintext
// add, plaintext multiply, ciphertext add/square, a vector-matrix product, an
// im2col convolution, and channel packing.  TenSEAL provided those primitives
// but its `auto_rescale` rewrites the ciphertext scale to `global_scale` after
// every rescale, which multiplies the cleartext by `global_scale / q` per
// multiplication (measured 1.174x per level on the paper profile).  This module
// exposes the same operations directly on SEAL, where `Ciphertext::scale()` is
// tracked exactly, so ciphertext inference keeps the 2^-21 fixed-point budget
// instead of accumulating a 17% per-layer gain error.
//
// Slot semantics are the ones the protocol already assumes (verified against
// TenSEAL 0.3.x, see seal/probes.py -> results/seal_backend/tenseal_semantics.json):
//   * a logical vector of length m occupies slots [0, m); slots >= m are zero
//   * im2col is tap-major: row k holds tap k of every window, so slot
//     k*windows + j is tap k of window j (tap rows padded to a power of two)
//   * conv2d_im2col maps window j to output slot j
//   * mm is an exact x @ M, implemented with the cyclic-diagonal identity
//   * pack_vectors concatenates the inputs in order
//
// Level policy (protocol SSOT section 5.3 counts one rescale per linear layer):
//   * add_plain / add_cipher: no rescale
//   * mul_plain, square, mm: one rescale for the whole op
//   * conv2d_im2col: no rescale, it returns the sum of at most k tap products so
//     its scale is global^2.  pack_vectors then consumes that one rescale (it
//     rescales once when the accumulated scale is far above global_scale).
//     Keeping the product un-rescaled is what lets the two ops share one level,
//     which is exactly the one-level-per-conv accounting of the protocol.
//
// Wire format (E4 byte compression, all opt-in; defaults reproduce the bytes
// recorded before 2026-10-29, i.e. uncompressed SEAL serialisation):
//   * compr = "none" | "zlib" | "zstd" selects SEAL's serialisation codec
//     explicitly.  The old code used SEAL's build-dependent default, so a
//     zstd-enabled build and a plain build produced 437,668 vs 917,625 bytes
//     for the same fresh ciphertext.
//   * encrypt_wire(..., symmetric=True) encrypts under the client's secret key
//     and serialises the seeded form (the second polynomial is replaced by its
//     PRNG seed), halving client->server ciphertexts.  Security: RLWE
//     symmetric encryption is IND-CPA, so the §11 simulator Enc(0) is
//     unchanged; only the client holds sk, so only M1 and ②_k can use it.
//   * public_context_bytes(seeded=True) sends pk / relin / Galois keys in
//     SEAL's seeded form (half size) - S_key is the client's largest cost.
//   * mod_switch_to_bits drops data primes the receiver does not need before
//     a server->client ciphertext (①_k, M3) is sent; encrypt_at() lets the
//     client encrypt ②_k at the chain index the remaining segment needs
//     (protocol §4.2 property 2: the p<->s coupling).
//
// Multiplication plaintexts are encoded at the ciphertext's *current* scale.
// SEAL tracks that scale exactly (it is 2^21 only on a fresh ciphertext and
// grows by 2^21/q_i on every rescale), and SEAL refuses to add ciphertexts
// whose scales differ, so every factor in one sum must share the scale of the
// ciphertext it multiplies.  Fresh encryptions that will be added to an
// existing ciphertext must use `encrypt_like`, which inherits both level and
// scale.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "seal/seal.h"
#include "seal/util/uintcore.h"

// SEAL >= 3.5 has the codec-aware save(), Serializable<> seeded objects and
// symmetric encryption; the SCI-bundled 3.3.2 has none of them.
#if defined(SEAL_VERSION_MAJOR) && \
    (SEAL_VERSION_MAJOR > 3 || (SEAL_VERSION_MAJOR == 3 && SEAL_VERSION_MINOR >= 5))
#define LITEPI_HAS_WIRE 1
#else
#define LITEPI_HAS_WIRE 0
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace seal;

#ifndef LITEPI_SEAL_VERSION
#define LITEPI_SEAL_VERSION "3.3.2"
#endif

namespace
{
using DArr = py::array_t<double, py::array::c_style | py::array::forcecast>;

std::vector<double> to_vector(const DArr &values)
{
    auto buf = values.request();
    const double *ptr = static_cast<const double *>(buf.ptr);
    return std::vector<double>(ptr, ptr + buf.size);
}

py::array_t<double> to_array(const std::vector<double> &values)
{
    py::array_t<double> out(values.size());
    if (!values.empty())
    {
        std::memcpy(out.request().ptr, values.data(), values.size() * sizeof(double));
    }
    return out;
}
}  // namespace

// A logical CKKS vector: the SEAL ciphertext plus the number of live slots.
class SealCtxt
{
public:
    SealCtxt() = default;
    SealCtxt(Ciphertext ct, std::size_t logical)
        : ct_(std::move(ct)), logical_(logical) {}

    std::shared_ptr<SealCtxt> clone() const
    {
        return std::make_shared<SealCtxt>(ct_, logical_);
    }

    Ciphertext ct_;
    std::size_t logical_ = 0;
};

// One party's CKKS context.  The client context holds the secret key; the
// server context is rebuilt from `public_context_bytes()` and never can decrypt.
class SealCtx
{
public:
    SealCtx(EncryptionParameters parms, double global_scale)
        : parms_(std::move(parms)),
          context_(std::make_shared<SEALContext>(parms_, true, sec_level_type::tc128)),
          global_scale_(global_scale),
          encoder_(std::make_unique<CKKSEncoder>(*context_)),
          evaluator_(std::make_unique<Evaluator>(*context_))
    {
    }

    // ---------------------------------------------------------------- helpers
    std::size_t slot_capacity() const
    {
        return parms_.poly_modulus_degree() / 2;
    }

    double global_scale() const
    {
        return global_scale_;
    }

    std::size_t chain_index(const Ciphertext &ct) const
    {
        return context_->get_context_data(ct.parms_id())->chain_index();
    }

    void align_levels(Ciphertext &a, Ciphertext &b) const
    {
        while (a.parms_id() != b.parms_id())
        {
            if (chain_index(a) > chain_index(b))
            {
                evaluator_->mod_switch_to_next_inplace(a);
            }
            else
            {
                evaluator_->mod_switch_to_next_inplace(b);
            }
        }
    }

    // out[j] = in[j + steps], cyclic over the slot ring ("rotation to the left").
    Ciphertext shift_left(const Ciphertext &in, int steps) const
    {
        Ciphertext out;
        evaluator_->rotate_vector(in, steps, galois_keys_, out);
        return out;
    }

    Ciphertext shift_right(const Ciphertext &in, int steps) const
    {
        return shift_left(in, -steps);
    }

    std::shared_ptr<SealCtxt> encrypt_values(const std::vector<double> &values,
                                             double scale) const
    {
        if (values.empty())
        {
            throw std::invalid_argument("cannot encrypt an empty vector");
        }
        Plaintext plain;
        Ciphertext ct;
        encoder_->encode(values, scale, plain);
        encryptor_->encrypt(plain, ct);
        return std::make_shared<SealCtxt>(std::move(ct), values.size());
    }

    std::shared_ptr<SealCtxt> encrypt_values(const std::vector<double> &values) const
    {
        return encrypt_values(values, global_scale_);
    }

    // Fresh encryption that can be added to `like`.  SEAL refuses additions of
    // ciphertexts whose scales differ, so the scale is inherited.  With
    // `match_level` the plaintext is encoded *at* the target level and
    // encrypted there, which reproduces the level without a mod switch (a mod
    // switch would divide the scale by the dropped primes and break the
    // equality again).  Protocol step 1 needs the level matched (step 5 does
    // not: the client's fresh share deliberately resets depth to the top).
    std::shared_ptr<SealCtxt> encrypt_like(const std::vector<double> &values,
                                           const Ciphertext &like,
                                           bool match_level = false) const
    {
        if (!match_level)
        {
            return encrypt_values(values, like.scale());
        }
        Plaintext plain;
        encode_at(values, like, plain);
        Ciphertext ct;
        encryptor_->encrypt(plain, ct);
        return std::make_shared<SealCtxt>(std::move(ct), values.size());
    }

    void encode_at(const std::vector<double> &values, const Ciphertext &like,
                   Plaintext &plain) const
    {
        // A one-element vector is the "constant plaintext" case that the
        // protocol uses for conv biases and scalar taps: SEAL's vector encode
        // would write it to slot 0 only and leave the remaining slots zero, so
        // route it through the scalar encode that fills every slot.
        if (values.size() == 1)
        {
            encoder_->encode(values[0], like.parms_id(), like.scale(), plain);
            return;
        }
        encoder_->encode(values, like.parms_id(), like.scale(), plain);
    }

    // Declared before the SEAL helpers so that the context exists when the
    // encoder and evaluator are constructed.
    EncryptionParameters parms_;
    std::shared_ptr<SEALContext> context_;
    double global_scale_ = 0.0;
    std::unique_ptr<CKKSEncoder> encoder_;
    std::unique_ptr<Evaluator> evaluator_;
    std::unique_ptr<Encryptor> encryptor_;
    std::unique_ptr<Decryptor> decryptor_;
    PublicKey public_key_;
    GaloisKeys galois_keys_;
    RelinKeys relin_keys_;
    SecretKey secret_key_;
    bool has_secret_ = false;
    // Client only: kept so that seeded (Serializable) keys can be emitted and
    // so that M1 / ②_k can be encrypted symmetrically.
    std::unique_ptr<KeyGenerator> keygen_;
    std::unique_ptr<Encryptor> sym_encryptor_;
    std::vector<int> galois_steps_;
    bool galois_all_ = true;
    // Client: relin / Galois keys are not kept; m0_relin / m0_galois generate
    // one key at a time while M0 is streamed (E2 client memory, plan A).
    bool lazy_eval_keys_ = false;
    // Server: Galois frames announced in the streamed M0 header / received.
    std::uint32_t galois_expected_ = 0;
    std::uint32_t galois_loaded_ = 0;

    // Galois elements the server's plan needs (all, or the ±2^i steps).
    std::vector<std::uint32_t> galois_elts() const
    {
        const auto *tool = context_->key_context_data()->galois_tool();
        auto elts = galois_all_ ? tool->get_elts_all()
                                : tool->get_elts_from_steps(galois_steps_);
        // get_elts_all lists 3^(N/4) twice (+N/4 and -N/4 are the same
        // rotation); the one-shot GaloisKeys object stores it once, so the
        // streamed form must send it once too.
        std::sort(elts.begin(), elts.end());
        elts.erase(std::unique(elts.begin(), elts.end()), elts.end());
        return elts;
    }

    // parms_id of a data level by chain index (top data level = largest).
    parms_id_type parms_id_at(int chain_index) const
    {
        auto data = context_->first_context_data();
        if (chain_index < 0 ||
            static_cast<std::size_t>(chain_index) > data->chain_index())
        {
            throw std::invalid_argument("chain index out of range");
        }
        while (data && data->chain_index() != static_cast<std::size_t>(chain_index))
        {
            data = data->next_context_data();
        }
        return data->parms_id();
    }

    int top_chain_index() const
    {
        return static_cast<int>(context_->first_context_data()->chain_index());
    }

    int modulus_bits(const parms_id_type &pid) const
    {
        int bits = 0;
        for (const auto &mod : context_->get_context_data(pid)->parms().coeff_modulus())
        {
            bits += util::get_significant_bit_count(mod.value());
        }
        return bits;
    }
};

#if LITEPI_HAS_WIRE
using wire_mode_t = compr_mode_type;
template <class T>
void save_obj(const T &obj, std::ostream &out, wire_mode_t mode)
{
    obj.save(out, mode);
}
#else
enum class wire_mode_t
{
    none
};
template <class T>
void save_obj(const T &obj, std::ostream &out, wire_mode_t)
{
    obj.save(out);
}
#endif

wire_mode_t parse_compr(const std::string &name)
{
    if (name == "none")
    {
        return wire_mode_t::none;
    }
#if LITEPI_HAS_WIRE && defined(SEAL_USE_ZLIB)
    if (name == "zlib")
    {
        return compr_mode_type::zlib;
    }
#endif
#if LITEPI_HAS_WIRE && defined(SEAL_USE_ZSTD)
    if (name == "zstd")
    {
        return compr_mode_type::zstd;
    }
#endif
    throw std::invalid_argument("compression mode '" + name +
                                "' is not available in this SEAL build");
}

std::vector<std::string> compr_modes()
{
    std::vector<std::string> out{"none"};
#if LITEPI_HAS_WIRE && defined(SEAL_USE_ZLIB)
    out.push_back("zlib");
#endif
#if LITEPI_HAS_WIRE && defined(SEAL_USE_ZSTD)
    out.push_back("zstd");
#endif
    return out;
}

// ------------------------------------------------------------------ context

// SEAL 4.x removed the static EncryptionParameters::Save/Load pair and made
// `load` an instance method that needs a matching SEALContext, which cannot be
// built before the parameters are known.  The profile is therefore written as
// (poly degree, coeff-modulus bit sizes): that is exactly what the protocol
// pins in SSOT section 6.1, so the blob stays version independent.
void write_parms(const EncryptionParameters &parms, std::ostream &out)
{
    std::uint64_t degree = parms.poly_modulus_degree();
    std::uint32_t nprimes =
        static_cast<std::uint32_t>(parms.coeff_modulus().size());
    out.write(reinterpret_cast<const char *>(&degree), sizeof(degree));
    out.write(reinterpret_cast<const char *>(&nprimes), sizeof(nprimes));
    for (const auto &mod : parms.coeff_modulus())
    {
        int bits = util::get_significant_bit_count(mod.value());
        out.write(reinterpret_cast<const char *>(&bits), sizeof(bits));
    }
}

EncryptionParameters read_parms(std::istream &in)
{
    std::uint64_t degree = 0;
    std::uint32_t nprimes = 0;
    in.read(reinterpret_cast<char *>(&degree), sizeof(degree));
    in.read(reinterpret_cast<char *>(&nprimes), sizeof(nprimes));
    std::vector<int> bits(nprimes);
    for (auto &b : bits)
    {
        in.read(reinterpret_cast<char *>(&b), sizeof(b));
    }
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(degree);
    parms.set_coeff_modulus(CoeffModulus::Create(degree, bits));
    return parms;
}

std::shared_ptr<SealCtx> new_context(std::size_t poly_modulus_degree,
                                     std::vector<int> coeff_mod_bit_sizes,
                                     int scale_bits, int max_rot_pow = -1,
                                     bool lazy_eval_keys = false)
{
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(
        CoeffModulus::Create(poly_modulus_degree, coeff_mod_bit_sizes));
    auto handle = std::make_shared<SealCtx>(std::move(parms),
                                            std::ldexp(1.0, scale_bits));
    handle->keygen_ = std::make_unique<KeyGenerator>(*handle->context_);
    KeyGenerator &keygen = *handle->keygen_;
    handle->secret_key_ = keygen.secret_key();
    handle->public_key_ = PublicKey();
    handle->relin_keys_ = RelinKeys();
    handle->galois_keys_ = GaloisKeys();
    keygen.create_public_key(handle->public_key_);
    handle->lazy_eval_keys_ = lazy_eval_keys;
    if (!lazy_eval_keys)
    {
        keygen.create_relin_keys(handle->relin_keys_);
    }
    if (max_rot_pow < 0 || static_cast<std::size_t>(max_rot_pow) + 1
                               >= static_cast<std::size_t>(poly_modulus_degree / 2))
    {
        if (!lazy_eval_keys)
        {
            keygen.create_galois_keys(handle->galois_keys_);
        }
        handle->galois_all_ = true;
    }
    else
    {
        // §6.4: 只生成 plan 实际用到的 ±2^i（i <= max_rot_pow）旋转步，其余
        // 非 2 幂位移由 SEAL 在这些元素上做 NAF 分解组合得到。SEAL 的
        // create_galois_keys(steps) 内部经 get_elts_from_steps/get_elt_from_step
        // 正确映射到 3^step（旧实现手写 2*step+1 的 Galois 元素是错误的，导致
        // rotate_vector 找不到 key）。
        std::vector<int> steps;
        steps.reserve(static_cast<std::size_t>(2 * (max_rot_pow + 1)));
        for (int i = 0; i <= max_rot_pow; ++i)
        {
            const int step = 1 << i;
            steps.push_back(step);
            steps.push_back(-step);
        }
        if (!lazy_eval_keys)
        {
            keygen.create_galois_keys(steps, handle->galois_keys_);
        }
        handle->galois_all_ = false;
        handle->galois_steps_ = steps;
    }
    handle->has_secret_ = true;
    handle->encryptor_ =
        std::make_unique<Encryptor>(*handle->context_, handle->public_key_);
#if LITEPI_HAS_WIRE
    handle->sym_encryptor_ =
        std::make_unique<Encryptor>(*handle->context_, handle->secret_key_);
#endif
    handle->decryptor_ =
        std::make_unique<Decryptor>(*handle->context_, handle->secret_key_);
    return handle;
}

// Serialized size of each key object plus the key counts.  E4 reports these as
// the client's key-distribution cost (protocol SSOT section 6.4 leaves the exact
// figures to the E4 measurement); the sizes are the SEAL wire encoding of the
// same objects the server loads, so they are comparable to the ciphertext bytes.
py::bytes public_context_bytes(const SealCtx &ctx);
py::bytes public_context_bytes(const SealCtx &ctx, const std::string &compr,
                               bool seeded);

py::dict key_sizes(const SealCtx &ctx)
{
    if (!ctx.has_secret_)
    {
        throw std::runtime_error("key_sizes needs the key-owning context");
    }
    if (ctx.lazy_eval_keys_)
    {
        throw std::runtime_error("key_sizes needs eagerly generated keys "
                                 "(new_context(lazy_eval_keys=False))");
    }
    // Uncompressed on every build (the E4 figures recorded so far); use
    // key_sizes_wire for the codec / seeded variants.
    auto size_of = [](const auto &obj)
    {
        std::ostringstream out(std::ios::binary);
        save_obj(obj, out, wire_mode_t::none);
        return static_cast<std::int64_t>(out.str().size());
    };
    py::dict out;
    out["public_key_bytes"] = size_of(ctx.public_key_);
    out["galois_keys_bytes"] = size_of(ctx.galois_keys_);
    out["galois_key_count"] = static_cast<std::int64_t>(ctx.galois_keys_.size());
    out["relin_keys_bytes"] = size_of(ctx.relin_keys_);
    out["secret_key_bytes"] = size_of(ctx.secret_key_);
    out["public_context_bytes"] = static_cast<std::int64_t>(public_context_bytes(ctx).cast<std::string>().size());
    return out;
}

py::bytes public_context_bytes(const SealCtx &ctx, const std::string &compr,
                               bool seeded)
{
    if (!ctx.has_secret_)
    {
        throw std::runtime_error("public_context_bytes needs the key-owning context");
    }
    if (ctx.lazy_eval_keys_ && !seeded)
    {
        throw std::runtime_error("lazy context: stream M0 with m0_header / "
                                 "m0_relin / m0_galois instead");
    }
    const auto mode = parse_compr(compr);
    std::ostringstream out(std::ios::binary);
    // The server needs the profile scale to encode its plaintext factors.
    out.write(reinterpret_cast<const char *>(&ctx.global_scale_),
              sizeof(ctx.global_scale_));
    write_parms(ctx.parms_, out);
    if (!seeded)
    {
        save_obj(ctx.public_key_, out, mode);
        save_obj(ctx.galois_keys_, out, mode);
        save_obj(ctx.relin_keys_, out, mode);
        return py::bytes(out.str());
    }
#if !LITEPI_HAS_WIRE
    throw std::runtime_error("seeded keys need SEAL >= 3.5 (build4.sh)");
#else
    // Seeded form: SEAL serialises a key's second polynomial as its PRNG seed.
    // The objects are freshly generated for the same secret key, which is all
    // the server needs (the client itself never rotates or relinearises).
    KeyGenerator &kg = *ctx.keygen_;
    kg.create_public_key().save(out, mode);
    if (ctx.galois_all_)
    {
        kg.create_galois_keys().save(out, mode);
    }
    else
    {
        kg.create_galois_keys(ctx.galois_steps_).save(out, mode);
    }
    kg.create_relin_keys().save(out, mode);
    return py::bytes(out.str());
#endif
}

py::bytes public_context_bytes(const SealCtx &ctx)
{
    return public_context_bytes(ctx, "none", false);
}

// E4: per-object key bytes under a given wire format (codec x seeded).
py::dict key_sizes_wire(const SealCtx &ctx, const std::string &compr, bool seeded)
{
    if (!ctx.has_secret_)
    {
        throw std::runtime_error("key_sizes_wire needs the key-owning context");
    }
    const auto mode = parse_compr(compr);
    auto size_of = [mode](const auto &obj)
    {
        std::ostringstream out(std::ios::binary);
        save_obj(obj, out, mode);
        return static_cast<std::int64_t>(out.str().size());
    };
    py::dict out;
    if (seeded)
    {
#if !LITEPI_HAS_WIRE
        throw std::runtime_error("seeded keys need SEAL >= 3.5 (build4.sh)");
#else
        KeyGenerator &kg = *ctx.keygen_;
        out["public_key_bytes"] = size_of(kg.create_public_key());
        out["galois_keys_bytes"] = ctx.galois_all_
            ? size_of(kg.create_galois_keys())
            : size_of(kg.create_galois_keys(ctx.galois_steps_));
        out["relin_keys_bytes"] = size_of(kg.create_relin_keys());
#endif
    }
    else
    {
        out["public_key_bytes"] = size_of(ctx.public_key_);
        out["galois_keys_bytes"] = size_of(ctx.galois_keys_);
        out["relin_keys_bytes"] = size_of(ctx.relin_keys_);
    }
    out["galois_key_count"] = static_cast<std::int64_t>(ctx.galois_keys_.size());
    out["compr"] = compr;
    out["seeded"] = seeded;
    out["public_context_bytes"] = static_cast<std::int64_t>(
        public_context_bytes(ctx, compr, seeded).cast<std::string>().size());
    return out;
}

std::shared_ptr<SealCtx> load_public_context(py::bytes blob)
{
    std::string raw = blob;
    std::istringstream in(raw, std::ios::binary);
    double global_scale = 0.0;
    in.read(reinterpret_cast<char *>(&global_scale), sizeof(global_scale));
    auto parms = read_parms(in);
    auto handle = std::make_shared<SealCtx>(std::move(parms), global_scale);
    handle->public_key_.load(*handle->context_, in);
    handle->galois_keys_.load(*handle->context_, in);
    handle->relin_keys_.load(*handle->context_, in);
    handle->encryptor_ =
        std::make_unique<Encryptor>(*handle->context_, handle->public_key_);
    handle->has_secret_ = false;
    return handle;
}

// ------------------------------------------------------- streamed M0 (plan A)
//
// The one-shot blob above needs every evaluation key in memory twice (the key
// objects and their serialisation): 757 MB peak / 241 MB steady on the client
// for 176 MiB of keys, although the client never rotates or relinearises.
// The streamed form sends M0 as frames
//     header  = scale | parms | pk | uint32 #galois
//     relin   = RelinKeys
//     galois  = GaloisKeys holding exactly one element      (x #galois)
// and a lazy client context generates each key right before it is
// serialised, so at most one key (~7 MB here) is alive at a time.  The
// server rebuilds the same GaloisKeys object with add_galois().
// Total bytes differ from the one-shot blob only by per-object headers.

py::bytes m0_header(const SealCtx &ctx, const std::string &compr, bool seeded)
{
    if (!ctx.has_secret_)
    {
        throw std::runtime_error("m0_header needs the key-owning context");
    }
    const auto mode = parse_compr(compr);
    std::ostringstream out(std::ios::binary);
    out.write(reinterpret_cast<const char *>(&ctx.global_scale_),
              sizeof(ctx.global_scale_));
    write_parms(ctx.parms_, out);
    if (seeded)
    {
#if !LITEPI_HAS_WIRE
        throw std::runtime_error("seeded keys need SEAL >= 3.5 (build4.sh)");
#else
        ctx.keygen_->create_public_key().save(out, mode);
#endif
    }
    else
    {
        save_obj(ctx.public_key_, out, mode);
    }
    std::uint32_t count = static_cast<std::uint32_t>(ctx.galois_elts().size());
    out.write(reinterpret_cast<const char *>(&count), sizeof(count));
    return py::bytes(out.str());
}

py::bytes m0_relin(const SealCtx &ctx, const std::string &compr, bool seeded)
{
    if (!ctx.has_secret_)
    {
        throw std::runtime_error("m0_relin needs the key-owning context");
    }
    const auto mode = parse_compr(compr);
    std::ostringstream out(std::ios::binary);
    if (seeded)
    {
#if !LITEPI_HAS_WIRE
        throw std::runtime_error("seeded keys need SEAL >= 3.5 (build4.sh)");
#else
        ctx.keygen_->create_relin_keys().save(out, mode);
#endif
    }
    else if (ctx.lazy_eval_keys_)
    {
        RelinKeys relin;
        ctx.keygen_->create_relin_keys(relin);
        save_obj(relin, out, mode);
    }
    else
    {
        save_obj(ctx.relin_keys_, out, mode);
    }
    return py::bytes(out.str());
}

py::bytes m0_galois(const SealCtx &ctx, std::uint32_t elt,
                    const std::string &compr, bool seeded)
{
    if (!ctx.has_secret_)
    {
        throw std::runtime_error("m0_galois needs the key-owning context");
    }
    const auto mode = parse_compr(compr);
    const std::vector<std::uint32_t> one{elt};
    std::ostringstream out(std::ios::binary);
    if (seeded)
    {
#if !LITEPI_HAS_WIRE
        throw std::runtime_error("seeded keys need SEAL >= 3.5 (build4.sh)");
#else
        ctx.keygen_->create_galois_keys(one).save(out, mode);
#endif
    }
    else
    {
        GaloisKeys g;
        ctx.keygen_->create_galois_keys(one, g);
        save_obj(g, out, mode);
    }
    return py::bytes(out.str());
}

// Server side of the streamed M0.
std::shared_ptr<SealCtx> load_public_header(py::bytes blob)
{
    std::string raw = blob;
    std::istringstream in(raw, std::ios::binary);
    double global_scale = 0.0;
    in.read(reinterpret_cast<char *>(&global_scale), sizeof(global_scale));
    auto parms = read_parms(in);
    auto handle = std::make_shared<SealCtx>(std::move(parms), global_scale);
    handle->public_key_.load(*handle->context_, in);
    std::uint32_t count = 0;
    in.read(reinterpret_cast<char *>(&count), sizeof(count));
    if (!in)
    {
        throw std::runtime_error("truncated M0 header");
    }
    handle->encryptor_ =
        std::make_unique<Encryptor>(*handle->context_, handle->public_key_);
    handle->has_secret_ = false;
    handle->galois_expected_ = count;
    return handle;
}

void load_relin(SealCtx &ctx, py::bytes blob)
{
    std::string raw = blob;
    std::istringstream in(raw, std::ios::binary);
    ctx.relin_keys_.load(*ctx.context_, in);
}

// Merge a one-element GaloisKeys object into the context's GaloisKeys.
void add_galois(SealCtx &ctx, py::bytes blob)
{
    std::string raw = blob;
    std::istringstream in(raw, std::ios::binary);
    GaloisKeys part;
    part.load(*ctx.context_, in);
    auto &dst = ctx.galois_keys_.data();
    auto &src = part.data();
    if (dst.empty())
    {
        dst.resize(src.size());
        ctx.galois_keys_.parms_id() = part.parms_id();
    }
    if (dst.size() != src.size() || ctx.galois_keys_.parms_id() != part.parms_id())
    {
        throw std::runtime_error("Galois key frame does not match the context");
    }
    for (std::size_t i = 0; i < src.size(); ++i)
    {
        if (!src[i].empty())
        {
            dst[i] = std::move(src[i]);
            ++ctx.galois_loaded_;
        }
    }
}

// Client: forget the evaluation keys of an eager context after M0 is sent
// (the memory returns to SEAL's pool; a lazy context never holds them).
void drop_eval_keys(SealCtx &ctx)
{
    ctx.galois_keys_ = GaloisKeys();
    ctx.relin_keys_ = RelinKeys();
}

// ------------------------------------------------------------------ encrypt

std::shared_ptr<SealCtxt> encrypt(const SealCtx &ctx, const DArr &values,
                                  double scale)
{
    if (!(scale > 0.0))
    {
        return ctx.encrypt_values(to_vector(values));
    }
    return ctx.encrypt_values(to_vector(values), scale);
}

std::shared_ptr<SealCtxt> encrypt_like(const SealCtx &ctx, const DArr &values,
                                       const SealCtxt &like, bool match_level)
{
    return ctx.encrypt_like(to_vector(values), like.ct_, match_level);
}

std::shared_ptr<SealCtxt> rotate(const SealCtx &ctx, const SealCtxt &ctxt,
                                 int steps)
{
    return std::make_shared<SealCtxt>(ctx.shift_left(ctxt.ct_, steps),
                                      ctxt.logical_);
}

py::array_t<double> decrypt(const SealCtx &ctx, const SealCtxt &ctxt)
{
    if (!ctx.has_secret_)
    {
        throw std::runtime_error("this context has no secret key");
    }
    Plaintext plain;
    ctx.decryptor_->decrypt(ctxt.ct_, plain);
    std::vector<double> decoded;
    ctx.encoder_->decode(plain, decoded);
    decoded.resize(std::min(ctxt.logical_, decoded.size()));
    return to_array(decoded);
}

int noise_budget(const SealCtx &ctx, const SealCtxt &ctxt)
{
    if (!ctx.has_secret_)
    {
        throw std::runtime_error("noise_budget needs the key-owning context");
    }
    return ctx.decryptor_->invariant_noise_budget(ctxt.ct_);
}

// --------------------------------------------------------- plaintext algebra

std::shared_ptr<SealCtxt> add_plain(const SealCtx &ctx, const SealCtxt &ctxt,
                                    const DArr &values)
{
    auto out = ctxt.clone();
    const auto vec = to_vector(values);
    if (vec.size() > ctx.slot_capacity())
    {
        throw std::invalid_argument("plaintext is longer than the slot count");
    }
    Plaintext plain;
    ctx.encode_at(vec, out->ct_, plain);
    ctx.evaluator_->add_plain_inplace(out->ct_, plain);
    return out;
}

std::shared_ptr<SealCtxt> add_plain_scalar(const SealCtx &ctx,
                                           const SealCtxt &ctxt, double value)
{
    auto out = ctxt.clone();
    Plaintext plain;
    ctx.encoder_->encode(value, out->ct_.parms_id(), out->ct_.scale(), plain);
    ctx.evaluator_->add_plain_inplace(out->ct_, plain);
    return out;
}

std::shared_ptr<SealCtxt> mul_plain(const SealCtx &ctx, const SealCtxt &ctxt,
                                    const DArr &values)
{
    auto out = ctxt.clone();
    const auto vec = to_vector(values);
    if (vec.size() > ctx.slot_capacity())
    {
        throw std::invalid_argument("plaintext is longer than the slot count");
    }
    Plaintext plain;
    ctx.encode_at(vec, out->ct_, plain);
    ctx.evaluator_->multiply_plain_inplace(out->ct_, plain);
    ctx.evaluator_->rescale_to_next_inplace(out->ct_);
    return out;
}

std::shared_ptr<SealCtxt> mul_plain_scalar(const SealCtx &ctx,
                                           const SealCtxt &ctxt, double value)
{
    if (value == 0.0)
    {
        // SEAL rejects transparent results, so a zero scale factor is realised
        // as a fresh encryption of zero.  Matching level and scale makes it
        // indistinguishable from a real leg, which is what the protocol's dummy
        // round needs, and it reveals nothing about the input.
        return ctx.encrypt_like(std::vector<double>(ctxt.logical_, 0.0),
                                ctxt.ct_, true);
    }
    auto out = ctxt.clone();
    Plaintext plain;
    ctx.encoder_->encode(value, out->ct_.parms_id(), out->ct_.scale(), plain);
    ctx.evaluator_->multiply_plain_inplace(out->ct_, plain);
    ctx.evaluator_->rescale_to_next_inplace(out->ct_);
    return out;
}

std::shared_ptr<SealCtxt> add_cipher(const SealCtx &ctx, const SealCtxt &a,
                                     const SealCtxt &b)
{
    if (a.logical_ != b.logical_)
    {
        throw std::invalid_argument("ciphertext sizes differ");
    }
    const double sa = a.ct_.scale();
    const double sb = b.ct_.scale();
    if (std::abs(sa - sb) > 1e-6 * std::max(sa, sb))
    {
        throw std::invalid_argument(
            "ciphertext scales differ (use encrypt_like for fresh operands)");
    }
    auto x = a.clone();
    auto y = b.clone();
    ctx.align_levels(x->ct_, y->ct_);
    ctx.evaluator_->add_inplace(x->ct_, y->ct_);
    return x;
}

std::shared_ptr<SealCtxt> square(const SealCtx &ctx, const SealCtxt &ctxt)
{
    auto out = ctxt.clone();
    ctx.evaluator_->square_inplace(out->ct_);
    ctx.evaluator_->relinearize_inplace(out->ct_, ctx.relin_keys_);
    ctx.evaluator_->rescale_to_next_inplace(out->ct_);
    return out;
}

// P2(x) = a0 + a1*x + a2*x^2 (protocol SSOT section 3.1), evaluated in Horner
// form so that every term of the single sum shares one scale:
//
//   inner = a2*x + a1     (plaintext multiplies, no rescale yet)
//   out   = Rescale(Rescale(Relin(x * inner))) + a0
//
// The ct x ct product must happen at the input level, so `inner` is neither
// mod-switched nor rescaled before it: the two rescales after the product take
// the triple scale back to the s * 1.17^2 neighbourhood the linear layers use.
// The square-only special case (a0 = a1 = 0, the poly-native network) is
// Rescale(Relin(x*x)) and costs one level, exactly as SSOT section 3.1 states.
// The general quadratic costs two, because SEAL charges a rescale for the
// constant multiplication that the section's accounting does not count.
std::shared_ptr<SealCtxt> poly2_eval(const SealCtx &ctx, const SealCtxt &ctxt,
                                     double a0, double a1, double a2)
{
    if (a0 == 0.0 && a1 == 0.0)
    {
        return square(ctx, ctxt);
    }
    Ciphertext inner = ctxt.ct_;
    {
        Plaintext p2;
        ctx.encoder_->encode(a2, inner.parms_id(), inner.scale(), p2);
        ctx.evaluator_->multiply_plain_inplace(inner, p2);
        Plaintext p1;
        ctx.encoder_->encode(a1, inner.parms_id(), inner.scale(), p1);
        ctx.evaluator_->add_plain_inplace(inner, p1);
    }

    auto out = ctxt.clone();
    ctx.evaluator_->multiply_inplace(out->ct_, inner);
    ctx.evaluator_->relinearize_inplace(out->ct_, ctx.relin_keys_);
    ctx.evaluator_->rescale_to_next_inplace(out->ct_);
    ctx.evaluator_->rescale_to_next_inplace(out->ct_);
    Plaintext p0;
    ctx.encoder_->encode(a0, out->ct_.parms_id(), out->ct_.scale(), p0);
    ctx.evaluator_->add_plain_inplace(out->ct_, p0);
    return out;
}

// ------------------------------------------------------------- linear layers

// Exact out[j] = sum_i in[i] * matrix[i][j] for a matrix of shape
// (logical slots, output width), with cyclic-diagonal rotations:
//
//   out[j] = sum_k in[(j + k) mod rows] * matrix[(j + k) mod rows][j]
//
// Rotation order matters for accuracy: a key-switch on a ciphertext at the
// 2^21 encoding scale injects about 3e5 coefficient units of noise, i.e. ~0.1
// in plaintext units, and that error would then be summed over the diagonals.
// The product is therefore formed *first* (multiply_plain lifts the scale to
// 2^42, so the same key-switch noise is ~7e-8 in plaintext units) and the
// rotation is applied to the product.  Linearity makes the two orders
// identical mathematically; only the noise differs.
//
// The cyclic wrap is covered by two half-sets of diagonals (forward r = j + k
// and backward r = j - k) instead of tiling the input, because tiling needs a
// rotation at the low scale and would reintroduce the same noise.
//
// All terms stay at the input level, sharing one scale, and are rescaled
// exactly once, as section 5.3 of the protocol requires for a linear layer.
std::shared_ptr<SealCtxt> mm(const SealCtx &ctx, const SealCtxt &ctxt,
                             const DArr &matrix)
{
    auto buf = matrix.request();
    if (buf.ndim != 2)
    {
        throw std::invalid_argument("mm expects a 2-D matrix");
    }
    const std::size_t rows = static_cast<std::size_t>(buf.shape[0]);
    const std::size_t cols = static_cast<std::size_t>(buf.shape[1]);
    if (rows != ctxt.logical_)
    {
        throw std::invalid_argument(
            "matrix rows must equal the ciphertext slot count");
    }
    if (cols == 0 || cols > rows)
    {
        throw std::invalid_argument("matrix width must be in [1, rows]");
    }
    const double *data = static_cast<const double *>(buf.ptr);
    py::gil_scoped_release release;

    std::vector<double> diagonal(rows, 0.0);
    Ciphertext acc;
    bool first = true;

    // With the product formed before the rotation, the weight for the term
    // out[j] += in[r] * W[r][j] (r = j + steps) has to sit at slot r, so the
    // diagonal is indexed by r and read as W[r][r - steps].
    // steps > 0: r = j + steps, j = r - steps >= 0
    // steps < 0: r = j + steps, j = r - steps < rows
    auto accumulate = [&](int steps) {
        bool nonzero = false;
        for (std::size_t r = 0; r < rows; ++r)
        {
            const long j = static_cast<long>(r) - steps;
            diagonal[r] = (j >= 0 && j < static_cast<long>(cols))
                              ? data[r * cols + static_cast<std::size_t>(j)]
                              : 0.0;
            nonzero = nonzero || (diagonal[r] != 0.0);
        }
        if (!nonzero)
        {
            // SEAL rejects transparent terms; a zero diagonal contributes nothing.
            return;
        }
        Plaintext plain;
        ctx.encode_at(diagonal, ctxt.ct_, plain);
        if (plain.is_zero())
        {
            // A diagonal whose entries are below the encoding resolution of this
            // level round to an all-zero plaintext; SEAL then rejects the
            // product as transparent.  Dropping the term is exact: it would
            // contribute zero after decoding anyway.
            return;
        }
        Ciphertext product;
        ctx.evaluator_->multiply_plain(ctxt.ct_, plain, product);
        Ciphertext moved =
            (steps == 0) ? std::move(product) : ctx.shift_left(product, steps);
        if (first)
        {
            acc = std::move(moved);
            first = false;
        }
        else
        {
            ctx.evaluator_->add_inplace(acc, moved);
        }
    };

    for (std::size_t k = 0; k < rows; ++k)
    {
        accumulate(static_cast<int>(k));
    }
    for (std::size_t k = 1; k < rows; ++k)
    {
        accumulate(-static_cast<int>(k));
    }
    if (first)
    {
        throw std::invalid_argument("mm matrix is identically zero");
    }
    ctx.evaluator_->rescale_to_next_inplace(acc);
    return std::make_shared<SealCtxt>(std::move(acc), cols);
}

// im2col is tap-major: slot k*windows + j is tap k of window j, so
//   out[j] = sum_k in[k*windows + j] * kernel[k]
// which is one rotation by k*windows per tap.  As in mm, the tap product is
// formed before the rotation so the key-switch noise is measured against the
// 2^42 product scale rather than the 2^21 encoding scale.
std::shared_ptr<SealCtxt> conv2d_im2col(const SealCtx &ctx, const SealCtxt &ctxt,
                                        const DArr &kernel, std::size_t windows)
{
    const auto tap = to_vector(kernel);
    if (windows == 0 || ctxt.logical_ % windows != 0)
    {
        throw std::invalid_argument(
            "slot count is not a multiple of the window count");
    }
    const std::size_t stride = ctxt.logical_ / windows;
    if (tap.size() > stride)
    {
        throw std::invalid_argument("kernel does not fit the im2col window");
    }
    py::gil_scoped_release release;
    const std::size_t slots =
        static_cast<std::size_t>(ctx.encoder_->slot_count());
    const std::size_t keep = std::min<std::size_t>(windows, slots);
    Ciphertext acc;
    bool first = true;
    for (std::size_t k = 0; k < tap.size(); ++k)
    {
        if (tap[k] == 0.0)
        {
            continue;
        }
        // The rotation moves slots k*windows..k*windows+windows-1 into the
        // valid output window, so place the tap there and leave everything
        // else zero.  That both applies the tap and suppresses the wrapped
        // sums for slots >= windows without an extra multiply or level.
        const std::size_t base = k * windows;
        std::vector<double> coeff(slots, 0.0);
        if (base < slots)
        {
            const std::size_t n = std::min<std::size_t>(keep, slots - base);
            std::fill(coeff.begin() + base, coeff.begin() + base + n, tap[k]);
        }
        Plaintext plain;
        ctx.encoder_->encode(coeff, ctxt.ct_.parms_id(), ctxt.ct_.scale(),
                             plain);
        Ciphertext product;
        ctx.evaluator_->multiply_plain(ctxt.ct_, plain, product);
        Ciphertext moved = (k == 0)
                               ? std::move(product)
                               : ctx.shift_left(product,
                                                static_cast<int>(k * windows));
        if (first)
        {
            acc = std::move(moved);
            first = false;
        }
        else
        {
            ctx.evaluator_->add_inplace(acc, moved);
        }
    }
    if (first)
    {
        throw std::invalid_argument("conv2d kernel is identically zero");
    }
    // The rescale is deliberately deferred to pack_vectors: rotating at the
    // squared scale keeps the key-switch noise ~1e-7 instead of ~1e-1, and the
    // single rescale charged to the pack keeps conv+pack at one level.
    return std::make_shared<SealCtxt>(std::move(acc), windows);
}

// Concatenate equal-length channel vectors.  The supports are disjoint, so this
// is a pure permutation of slots; the single rescale that pack takes is the
// deferred one from conv2d_im2col, so conv+pack still cost exactly one level.
std::shared_ptr<SealCtxt> pack_vectors(
    const SealCtx &ctx, const std::vector<std::shared_ptr<SealCtxt>> &vectors)
{
    if (vectors.empty())
    {
        throw std::invalid_argument("pack_vectors needs at least one ciphertext");
    }
    const std::size_t width = vectors.front()->logical_;
    for (const auto &v : vectors)
    {
        if (v->logical_ != width)
        {
            throw std::invalid_argument("pack_vectors needs equal-sized inputs");
        }
    }
    const std::size_t total = width * vectors.size();
    if (total > ctx.slot_capacity())
    {
        throw std::invalid_argument("packed vector exceeds the slot count");
    }
    py::gil_scoped_release release;
    Ciphertext acc = vectors.front()->ct_;
    for (std::size_t i = 1; i < vectors.size(); ++i)
    {
        Ciphertext shifted =
            ctx.shift_right(vectors[i]->ct_, static_cast<int>(i * width));
        ctx.evaluator_->add_inplace(acc, shifted);
    }
    // At the squared scale (conv output) the accumulated rotation noise is
    // negligible; one rescale here returns to the global scale and pays the
    // level that conv deferred.
    if (acc.scale() > 4.0 * ctx.global_scale())
    {
        ctx.evaluator_->rescale_to_next_inplace(acc);
    }
    return std::make_shared<SealCtxt>(std::move(acc), total);
}

// ------------------------------------------------------------ client encoder

// Build the tap-major im2col layout and encrypt it in one shot.  Mirrors
// TenSEAL's `im2col_encoding` slot order: the tap rows are padded up to the
// next power of two and every tap row stores all windows.
std::pair<std::vector<double>, std::size_t> im2col_layout(
    const SealCtx &ctx, const DArr &image, int kernel_rows, int kernel_cols,
    int stride, int stride_cols)
{
    auto buf = image.request();
    if (buf.ndim != 2)
    {
        throw std::invalid_argument("im2col input must be a 2-D image");
    }
    const std::size_t height = static_cast<std::size_t>(buf.shape[0]);
    const std::size_t width = static_cast<std::size_t>(buf.shape[1]);
    if (kernel_rows <= 0 || kernel_cols <= 0 || stride <= 0 || stride_cols < 0)
    {
        throw std::invalid_argument("kernel and stride must be positive");
    }
    const std::size_t kh = static_cast<std::size_t>(kernel_rows);
    const std::size_t kw = static_cast<std::size_t>(kernel_cols);
    const std::size_t step = static_cast<std::size_t>(stride);
    const std::size_t step_c = stride_cols > 0 ? static_cast<std::size_t>(stride_cols) : step;
    if (kh > height || kw > width)
    {
        throw std::invalid_argument("kernel is larger than the image");
    }
    const std::size_t taps = kh * kw;
    std::size_t padded = 1;
    while (padded < taps)
    {
        padded <<= 1;
    }
    const std::size_t rows = (height - kh) / step + 1;
    const std::size_t cols = (width - kw) / step_c + 1;
    const std::size_t windows = rows * cols;
    if (windows * padded > ctx.slot_capacity())
    {
        throw std::invalid_argument("im2col encoding exceeds the slot count");
    }
    const double *src = static_cast<const double *>(buf.ptr);
    std::vector<double> flat(windows * padded, 0.0);
    for (std::size_t r = 0; r < rows; ++r)
    {
        for (std::size_t c = 0; c < cols; ++c)
        {
            const std::size_t window = r * cols + c;
            for (std::size_t i = 0; i < kh; ++i)
            {
                for (std::size_t j = 0; j < kw; ++j)
                {
                    flat[(i * kw + j) * windows + window] =
                        src[(r * step + i) * width + (c * step_c + j)];
                }
            }
        }
    }
    return {std::move(flat), windows};
}

std::pair<std::shared_ptr<SealCtxt>, std::size_t> im2col_encrypt(
    SealCtx &ctx, const DArr &image, int kernel_rows, int kernel_cols, int stride,
    int stride_cols)
{
    auto layout = im2col_layout(ctx, image, kernel_rows, kernel_cols, stride,
                                stride_cols);
    return {ctx.encrypt_values(layout.first), layout.second};
}

std::pair<py::array_t<double>, std::size_t> im2col_flat(
    SealCtx &ctx, const DArr &image, int kernel_rows, int kernel_cols, int stride,
    int stride_cols)
{
    auto layout = im2col_layout(ctx, image, kernel_rows, kernel_cols, stride,
                                stride_cols);
    return {to_array(layout.first), layout.second};
}

// ------------------------------------------------- multi-channel conv (E1 deep)
//
// Row-blocked matrix product for convolutions whose input the client
// re-encodes in im2col form after every hybrid layer (deep networks).
// Layout: a ciphertext holds R = slots / windows rows of `windows` slots,
// row t = im2col tap t (input channel, kernel offset) over all windows.  The
// input is a list of such blocks (taps beyond the last block are zero).  The
// output has the same layout with one row per output channel, i.e. the
// (channel, oh, ow) order of a PyTorch flatten, split over ceil(O / R)
// ciphertexts.
//
// For each (output block c, input block b) the R x R sub-matrix is applied
// with the diagonal method, multiply-before-rotate:
//     out_c += sum_d rot_{d*W}( x_b * rot_{-d*W}(diag_d) )
// Rotations happen on the products, at the squared scale: on this profile a
// rotation at the global scale 2^21 costs 0.06-0.3 absolute error (40-bit
// special prime), at 2^42 it is negligible.  One rescale at the end, so the
// whole convolution costs one level, like conv2d_im2col + pack_vectors.
std::vector<std::shared_ptr<SealCtxt>> matmul_rows(
    const SealCtx &ctx, const std::vector<std::shared_ptr<SealCtxt>> &blocks,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &matrix,
    std::size_t windows)
{
    auto buf = matrix.request();
    if (buf.ndim != 2)
    {
        throw std::invalid_argument("matmul_rows expects a 2-D matrix");
    }
    const std::size_t slots = ctx.slot_capacity();
    if (windows == 0 || slots % windows != 0)
    {
        throw std::invalid_argument("windows must divide the slot count");
    }
    const std::size_t rows = slots / windows;
    const std::size_t out_rows = static_cast<std::size_t>(buf.shape[0]);
    const std::size_t in_rows = static_cast<std::size_t>(buf.shape[1]);
    if (blocks.empty() || in_rows > blocks.size() * rows)
    {
        throw std::invalid_argument("matrix has more input rows than the blocks");
    }
    const double *m = static_cast<const double *>(buf.ptr);
    const std::size_t out_blocks = (out_rows + rows - 1) / rows;
    std::vector<std::shared_ptr<SealCtxt>> outs;
    py::gil_scoped_release release;
    for (std::size_t c = 0; c < out_blocks; ++c)
    {
        Ciphertext acc;
        bool first = true;
        for (std::size_t b = 0; b < blocks.size(); ++b)
        {
            const Ciphertext &x = blocks[b]->ct_;
            for (std::size_t d = 0; d < rows; ++d)
            {
                // p[s] = diag_d[s - d*W]: row i of diag_d is M[cR+i, bR+(i+d)%R]
                std::vector<double> p(slots, 0.0);
                bool any = false;
                for (std::size_t i = 0; i < rows; ++i)
                {
                    const std::size_t orow = c * rows + i;
                    const std::size_t irow = b * rows + (i + d) % rows;
                    if (orow >= out_rows || irow >= in_rows)
                    {
                        continue;
                    }
                    const double w = m[orow * in_rows + irow];
                    if (w == 0.0)
                    {
                        continue;
                    }
                    any = true;
                    const std::size_t dst = ((i + d) % rows) * windows;
                    std::fill(p.begin() + dst, p.begin() + dst + windows, w);
                }
                if (!any)
                {
                    continue;
                }
                Plaintext plain;
                ctx.encoder_->encode(p, x.parms_id(), x.scale(), plain);
                Ciphertext prod;
                ctx.evaluator_->multiply_plain(x, plain, prod);
                if (d != 0)
                {
                    prod = ctx.shift_left(prod, static_cast<int>(d * windows));
                }
                if (first)
                {
                    acc = std::move(prod);
                    first = false;
                }
                else
                {
                    ctx.evaluator_->add_inplace(acc, prod);
                }
            }
        }
        if (first)
        {
            throw std::invalid_argument("matmul_rows: an output block is all zero");
        }
        ctx.evaluator_->rescale_to_next_inplace(acc);
        const std::size_t live = std::min(rows, out_rows - c * rows);
        outs.push_back(std::make_shared<SealCtxt>(std::move(acc), live * windows));
    }
    return outs;
}

// -------------------------------------------------------------- wire format

py::bytes serialize(const SealCtxt &ctxt, const std::string &compr = "none")
{
    std::ostringstream out(std::ios::binary);
    const std::uint64_t logical = ctxt.logical_;
    out.write(reinterpret_cast<const char *>(&logical), sizeof(logical));
    save_obj(ctxt.ct_, out, parse_compr(compr));
    return py::bytes(out.str());
}

// Encrypt and serialise in one step, optionally under the secret key in
// SEAL's seeded form.  chain_index < 0 means the top data level.
py::bytes encrypt_wire(const SealCtx &ctx, const DArr &values, double scale,
                       int chain_index, bool symmetric, const std::string &compr)
{
    const auto vec = to_vector(values);
    if (vec.empty())
    {
        throw std::invalid_argument("cannot encrypt an empty vector");
    }
    if (symmetric && !ctx.has_secret_)
    {
        throw std::runtime_error("symmetric encryption needs the secret key");
    }
    const auto mode = parse_compr(compr);
    const auto pid = chain_index < 0 ? ctx.context_->first_parms_id()
                                     : ctx.parms_id_at(chain_index);
    Plaintext plain;
    ctx.encoder_->encode(vec, pid, scale > 0.0 ? scale : ctx.global_scale_, plain);
    std::ostringstream out(std::ios::binary);
    const std::uint64_t logical = vec.size();
    out.write(reinterpret_cast<const char *>(&logical), sizeof(logical));
    if (symmetric)
    {
#if LITEPI_HAS_WIRE
        ctx.sym_encryptor_->encrypt_symmetric(plain).save(out, mode);
#else
        throw std::runtime_error("symmetric encryption needs SEAL >= 3.5");
#endif
    }
    else
    {
        Ciphertext ct;
        ctx.encryptor_->encrypt(plain, ct);
        save_obj(ct, out, mode);
    }
    return py::bytes(out.str());
}

// Public-key encryption at a given chain index (short-chain ②_k / tests).
std::shared_ptr<SealCtxt> encrypt_at(const SealCtx &ctx, const DArr &values,
                                     int chain_index, double scale)
{
    const auto vec = to_vector(values);
    Plaintext plain;
    ctx.encoder_->encode(vec, ctx.parms_id_at(chain_index),
                         scale > 0.0 ? scale : ctx.global_scale_, plain);
    Ciphertext ct;
    ctx.encryptor_->encrypt(plain, ct);
    return std::make_shared<SealCtxt>(std::move(ct), vec.size());
}

// Drop data primes while the remaining modulus keeps >= needed_bits.  Plain
// modulus switching (no rescale) leaves message and scale unchanged, so the
// receiver decrypts the same values from a smaller ciphertext.
std::shared_ptr<SealCtxt> mod_switch_to_bits(const SealCtx &ctx,
                                             const SealCtxt &ctxt,
                                             int needed_bits)
{
    auto out = ctxt.clone();
    while (true)
    {
        auto data = ctx.context_->get_context_data(out->ct_.parms_id());
        auto next = data->next_context_data();
        if (!next || ctx.modulus_bits(next->parms_id()) < needed_bits)
        {
            break;
        }
        ctx.evaluator_->mod_switch_to_next_inplace(out->ct_);
    }
    return out;
}

std::vector<int> chain_prime_bits(const SealCtx &ctx)
{
    // Data primes from the bottom (chain index 0 keeps only the first).
    std::vector<int> bits;
    for (const auto &mod : ctx.context_->first_context_data()->parms().coeff_modulus())
    {
        bits.push_back(util::get_significant_bit_count(mod.value()));
    }
    return bits;
}

std::shared_ptr<SealCtxt> deserialize(const SealCtx &ctx, py::bytes blob)
{
    std::string raw = blob;
    std::istringstream in(raw, std::ios::binary);
    std::uint64_t logical = 0;
    in.read(reinterpret_cast<char *>(&logical), sizeof(logical));
    Ciphertext ct;
    ct.load(*ctx.context_, in);
    return std::make_shared<SealCtxt>(std::move(ct),
                                      static_cast<std::size_t>(logical));
}

PYBIND11_MODULE(litepi_seal, m)
{
    m.doc() = "LitePI HE backend: CKKS on SEAL (3.3.2 or 4.1.x)";
    m.attr("SEAL_VERSION") = LITEPI_SEAL_VERSION;

    py::class_<SealCtxt, std::shared_ptr<SealCtxt>>(m, "Ciphertext")
        .def_property_readonly("size", [](const SealCtxt &c) {
            return c.logical_;
        })
        .def("serialize", [](const SealCtxt &c, const std::string &compr) {
            return serialize(c, compr);
        }, py::arg("compr") = "none");

    py::class_<SealCtx, std::shared_ptr<SealCtx>>(m, "SealContext")
        .def_property_readonly("has_secret", [](const SealCtx &c) {
            return c.has_secret_;
        })
        .def("slot_capacity", &SealCtx::slot_capacity)
        .def("index", [](const SealCtx &c, const SealCtxt &x) {
            return c.chain_index(x.ct_);
        })
        .def("scale", [](const SealCtx &c, const SealCtxt &x) {
            return x.ct_.scale();
        })
        .def("encrypt", &encrypt, py::arg("values"), py::arg("scale") = 0.0)
        .def("encrypt_like", &encrypt_like, py::arg("values"), py::arg("like"),
             py::arg("match_level") = false)
        .def("rotate", &rotate, py::arg("ctxt"), py::arg("steps"))
        .def("decrypt", &decrypt)
        .def("noise_budget", &noise_budget)
        .def("add_plain", &add_plain)
        .def("add_plain_scalar", &add_plain_scalar)
        .def("mul_plain", &mul_plain)
        .def("mul_plain_scalar", &mul_plain_scalar)
        .def("add_cipher", &add_cipher)
        .def("square", &square)
        .def("poly2_eval", &poly2_eval, py::arg("ctxt"), py::arg("a0"),
             py::arg("a1"), py::arg("a2"))
        .def("mm", &mm)
        .def("conv2d_im2col", &conv2d_im2col)
        .def("pack_vectors", &pack_vectors)
        .def("matmul_rows", &matmul_rows, py::arg("blocks"), py::arg("matrix"),
             py::arg("windows"))
        .def("im2col_encrypt", &im2col_encrypt,
             py::arg("image"), py::arg("kernel_rows"),
             py::arg("kernel_cols"), py::arg("stride"), py::arg("stride_cols") = 0)
        .def("deserialize", &deserialize)
        .def("im2col_flat", &im2col_flat,
             py::arg("image"), py::arg("kernel_rows"),
             py::arg("kernel_cols"), py::arg("stride"), py::arg("stride_cols") = 0)
        .def("encrypt_wire", &encrypt_wire, py::arg("values"),
             py::arg("scale") = 0.0, py::arg("chain_index") = -1,
             py::arg("symmetric") = false, py::arg("compr") = "none")
        .def("encrypt_at", &encrypt_at, py::arg("values"),
             py::arg("chain_index"), py::arg("scale") = 0.0)
        .def("mod_switch_to_bits", &mod_switch_to_bits, py::arg("ctxt"),
             py::arg("needed_bits"))
        .def("modulus_bits", [](const SealCtx &c, const SealCtxt &x) {
            return c.modulus_bits(x.ct_.parms_id());
        })
        .def("top_chain_index", &SealCtx::top_chain_index)
        .def("galois_elts", &SealCtx::galois_elts)
        .def("m0_header", &m0_header, py::arg("compr") = "none",
             py::arg("seeded") = false)
        .def("m0_relin", &m0_relin, py::arg("compr") = "none",
             py::arg("seeded") = false)
        .def("m0_galois", &m0_galois, py::arg("elt"), py::arg("compr") = "none",
             py::arg("seeded") = false)
        .def("load_relin", &load_relin)
        .def("add_galois", &add_galois)
        .def("drop_eval_keys", &drop_eval_keys)
        .def_property_readonly("lazy_eval_keys", [](const SealCtx &c) {
            return c.lazy_eval_keys_;
        })
        .def_property_readonly("galois_expected", [](const SealCtx &c) {
            return c.galois_expected_;
        })
        .def_property_readonly("galois_loaded", [](const SealCtx &c) {
            return c.galois_loaded_;
        })
        .def("chain_prime_bits", &chain_prime_bits)
        .def("chain_primes", [](const SealCtx &c) {
            std::vector<std::uint64_t> out;
            for (const auto &mod :
                 c.context_->first_context_data()->parms().coeff_modulus())
            {
                out.push_back(mod.value());
            }
            return out;
        });

    m.def("new_context", &new_context, py::arg("poly_modulus_degree"),
          py::arg("coeff_mod_bit_sizes"), py::arg("scale_bits"),
          py::arg("max_rot_pow") = -1, py::arg("lazy_eval_keys") = false);
    m.def("load_public_header", &load_public_header);
    m.def("public_context_bytes",
          py::overload_cast<const SealCtx &, const std::string &, bool>(
              &public_context_bytes),
          py::arg("ctx"), py::arg("compr") = "none", py::arg("seeded") = false);
    m.def("key_sizes", &key_sizes);
    m.def("key_sizes_wire", &key_sizes_wire, py::arg("ctx"),
          py::arg("compr") = "none", py::arg("seeded") = false);
    m.def("compr_modes", &compr_modes);
    m.def("load_public_context", &load_public_context);
    m.def("slot_count", [](const SealCtxt &x) { return x.logical_; });
}
