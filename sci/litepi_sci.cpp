// litepi_sci.cpp - pybind11 wrapper for SCI (CrypTFlow2) ReLU protocol.
//
// Exposes a two-party secure ReLU to Python. Each party (server/client)
// holds an additive share of the input and receives an additive share of
// the output. CrypTFlow2's millionaire protocol uses OT extension; constant
// round complexity does NOT mean one end-to-end RTT. relu_real adds one
// BOB-to-ALICE masked resharing message for the CKKS real-additive bridge.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <cstdint>
#include <cmath>
#include <limits>
#include <openssl/rand.h>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

#include "BuildingBlocks/aux-protocols.h"
#include "Math/math-functions.h"

namespace py = pybind11;
using namespace sci;

class SecureReLU {
public:
  // party: 1 = ALICE (server), 2 = BOB (client)
  // port: base port for the OT/IO channels
  // address: server IP (client connects to it)
  SecureReLU(int party, int port, const std::string &address)
      : party_(party), port_(port), address_(address) {
    iopack_ = new IOPack(party, port, address);
    otpack_ = new OTPack(iopack_, party);
    math_ = new MathFunctions(party, iopack_, otpack_);
    aux_ = new AuxProtocols(party, iopack_, otpack_);
  }

  ~SecureReLU() {
    delete aux_;
    delete math_;
    delete otpack_;
    delete iopack_;
  }

  // Secure ReLU over an array of fixed-point shares.
  // Input/output are uint64 arrays representing two's-complement shares.
  // bw: bit-width of the fixed-point representation.
  py::array_t<uint64_t> relu(py::array_t<uint64_t> shares, int bw) {
    py::buffer_info buf = shares.request();
    if (bw < 2 || bw > 64 || buf.ndim != 1 || buf.shape[0] <= 0 ||
        buf.shape[0] > std::numeric_limits<int>::max() / 8 ||
        buf.strides[0] != static_cast<py::ssize_t>(sizeof(uint64_t))) {
      throw py::value_error("require nonempty contiguous uint64 vector and 2 <= bw <= 64");
    }
    int n = static_cast<int>(buf.shape[0]);

    auto result = py::array_t<uint64_t>(n);
    py::buffer_info rbuf = result.request();

    uint64_t *in = static_cast<uint64_t *>(buf.ptr);
    uint64_t *out = static_cast<uint64_t *>(rbuf.ptr);

    // Both parties run inside one process (server thread + client thread in
    // loopback).  The C++ call blocks on network I/O, so the GIL must be
    // released or the two parties deadlock waiting for each other.
    {
      py::gil_scoped_release release;
      // six = 0 means standard ReLU (not ReLU6).
      math_->ReLU(n, in, out, bw, 0);
    }

    return result;
  }

  // Shares of the most significant bit (sign) of a fixed-point vector.
  // Output is a boolean share per element: one party's share is 1 when the
  // reconstructed value is negative.  This is the DReLU primitive used by the
  // secure argmax tournament.
  py::array_t<uint8_t> drelu(py::array_t<uint64_t> shares, int bw) {
    py::buffer_info buf = shares.request();
    int n = static_cast<int>(buf.shape[0]);

    auto result = py::array_t<uint8_t>(n);
    py::buffer_info rbuf = result.request();

    uint64_t *in = static_cast<uint64_t *>(buf.ptr);
    uint8_t *out = static_cast<uint8_t *>(rbuf.ptr);

    {
      py::gil_scoped_release release;
      aux_->MSB(in, out, n, bw);
    }
    return result;
  }

  // y = sel * x for arithmetic x and boolean sel, all secret-shared.
  // Used to select the champion score and index inside the tournament.
  py::array_t<uint64_t> mux(py::array_t<uint8_t> sel,
                            py::array_t<uint64_t> x, int bw_x, int bw_y) {
    py::buffer_info sbuf = sel.request();
    py::buffer_info xbuf = x.request();
    int n = static_cast<int>(xbuf.shape[0]);
    if (sbuf.shape[0] != n)
      throw std::runtime_error("mux: sel and x must have the same length");

    auto result = py::array_t<uint64_t>(n);
    py::buffer_info rbuf = result.request();

    uint8_t *s = static_cast<uint8_t *>(sbuf.ptr);
    uint64_t *xp = static_cast<uint64_t *>(xbuf.ptr);
    uint64_t *out = static_cast<uint64_t *>(rbuf.ptr);

    {
      py::gil_scoped_release release;
      aux_->multiplexer(s, xp, out, n, bw_x, bw_y);
    }
    return result;
  }

  // Return REAL-additive ReLU shares, not independently centered ring shares.
  // For signed k-bit ReLU, 0 <= y < H=2^(k-1). BOB draws rho in [0,H),
  // sends (y_BOB-rho) mod 2^k. ALICE centers the sum with y_ALICE to y-rho.
  // Since -H < y-rho < H, the two real shares sum exactly to y/2^f.
  // ALICE learns only a masked output, never rho or the unmasked activation.
  // This is statistical masking, NOT a perfect OTP or a circuit-privacy proof;
  // its hiding bound additionally requires an a priori bound on y.
  py::array_t<double> relu_real(py::array_t<uint64_t> shares, int bw, int frac) {
    if (bw < 2 || bw > 53 || frac < 0 || frac >= bw) {
      throw py::value_error("real bridge requires 2 <= bw <= 53 and 0 <= frac < bw");
    }
    auto ring = relu(shares, bw);
    auto rbuf = ring.request();
    int n = static_cast<int>(rbuf.shape[0]);
    auto *local = static_cast<uint64_t *>(rbuf.ptr);
    auto result = py::array_t<double>(n);
    auto obuf = result.request();
    auto *out = static_cast<double *>(obuf.ptr);
    const uint64_t half = uint64_t{1} << (bw - 1);
    const uint64_t modulus = uint64_t{1} << bw;
    const uint64_t mask = modulus - 1;
    const double scale = std::ldexp(1.0, frac);
    std::vector<uint64_t> message(n), rho(n);
    {
      py::gil_scoped_release release;
      if (party_ == BOB) {
        // OpenSSL's CSPRNG avoids unchecked CPU-specific seed instructions here.
        if (RAND_bytes(reinterpret_cast<unsigned char *>(rho.data()), n * 8) != 1) {
          throw std::runtime_error("CSPRNG failed during ReLU output resharing");
        }
        for (int i = 0; i < n; ++i) {
          rho[i] &= half - 1;
          message[i] = (local[i] - rho[i]) & mask;
          out[i] = static_cast<double>(rho[i]) / scale;
        }
        iopack_->io->send_data(message.data(), n * 8);
        iopack_->io->flush();
      } else {
        iopack_->io->recv_data(message.data(), n * 8);
        for (int i = 0; i < n; ++i) {
          const uint64_t value = (local[i] + message[i]) & mask;
          const int64_t centered = value >= half
              ? static_cast<int64_t>(value) - static_cast<int64_t>(modulus)
              : static_cast<int64_t>(value);
          out[i] = static_cast<double>(centered) / scale;
        }
      }
    }
    return result;
  }

  // Communication bytes sent since construction (for profiling).
  uint64_t bytes_sent() { return iopack_->get_comm(); }

private:
  int party_;
  int port_;
  std::string address_;
  IOPack *iopack_;
  OTPack *otpack_;
  MathFunctions *math_;
  AuxProtocols *aux_;
};

PYBIND11_MODULE(litepi_sci, m) {
  m.doc() = "LitePI MPC backend: CrypTFlow2 (SCI) secure ReLU";

  py::class_<SecureReLU>(m, "SecureReLU")
      .def(py::init([](int party, int port, const std::string &address) {
             // Construction runs the TCP handshake, base OTs and IKNP
             // extension, all of which block on the network.  Release the
             // GIL so the other party's thread (same process, loopback) can
             // run concurrently.
             py::gil_scoped_release release;
             return std::unique_ptr<SecureReLU>(
                 new SecureReLU(party, port, address));
           }),
           py::arg("party"), py::arg("port"), py::arg("address"))
      .def("relu", &SecureReLU::relu,
           py::arg("shares"), py::arg("bw"))
      .def("drelu", &SecureReLU::drelu,
           py::arg("shares"), py::arg("bw"))
      .def("mux", &SecureReLU::mux,
           py::arg("sel"), py::arg("x"), py::arg("bw_x"), py::arg("bw_y"))
      .def("relu_real", &SecureReLU::relu_real,
           py::arg("shares"), py::arg("bw"), py::arg("frac"))
      .def("bytes_sent", &SecureReLU::bytes_sent);

  // Adapter checks this before constructing a session; old binaries fail closed.
  m.attr("REAL_BRIDGE_VERSION") = 1;
  m.attr("ALICE") = 1;
  m.attr("BOB") = 2;
}
