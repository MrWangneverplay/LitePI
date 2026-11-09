// pybind11 wrapper around the EzPC/SCI (CrypTFlow2) ring-ReLU protocol.
//
// The Python reference path (analysis/secure_relu_net.py) talks the same
// two-party interface but implements the comparison with P-256 Simplest-OT.
// This module keeps that interface and runs the SCI constant-round
// Millionaire/OT-extension based comparison instead.
//
// Usage from Python (two processes, party 1 = ALICE/server, party 2 = BOB/client):
//
//     sess = litepi_sci.SecureReLU(party, port, "127.0.0.1")
//     out  = sess.relu(share_ring, bw)          # uint64 two's-complement shares
//     sess.bytes_sent(); sess.rounds()

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "Math/math-functions.h"
#include "OT/ot_pack.h"
#include "utils/io_pack.h"

namespace py = pybind11;

class SecureReLU {
public:
  SecureReLU(int party, int port, const std::string &address)
      : party_(party), port_(port), address_(address) {
    if (party_ != 1 && party_ != 2) {
      throw std::invalid_argument("party must be 1 (ALICE/server) or 2 (BOB/client)");
    }
    // The IOPack constructor blocks on accept()/connect() and OTPack performs
    // the base-OT setup, so the GIL must be released for the other party (in a
    // separate Python thread) to make progress.
    py::gil_scoped_release release;
    iopack_ = new sci::IOPack(party_, port_, address_);
    otpack_ = new sci::OTPack(iopack_, party_);
    math_ = new MathFunctions(party_, iopack_, otpack_);
  }

  ~SecureReLU() {
    delete math_;
    delete otpack_;
    delete iopack_;
  }

  py::array_t<uint64_t> relu(
      py::array_t<uint64_t, py::array::c_style | py::array::forcecast> shares,
      int bw) {
    if (bw <= 1 || bw > 64) {
      throw std::invalid_argument("ring bit width must be in [2, 64]");
    }
    const auto info = shares.request();
    const int64_t n = static_cast<int64_t>(info.size);
    auto out = py::array_t<uint64_t>(n);
    if (n == 0) {
      return out;
    }

    std::vector<uint64_t> in(static_cast<size_t>(n));
    std::memcpy(in.data(), info.ptr, static_cast<size_t>(n) * sizeof(uint64_t));
    uint64_t *out_ptr = static_cast<uint64_t *>(out.request().ptr);

    py::gil_scoped_release release;
    math_->ReLU(static_cast<int32_t>(n), in.data(), out_ptr, bw, 0);
    return out;
  }

  // Total communication (bytes) over the SCI channels, including base OTs.
  uint64_t bytes_sent() const { return iopack_->get_comm(); }

  // Round counter from SCI's own socket layer (io + GC channel).
  uint64_t rounds() const { return iopack_->get_rounds(); }

private:
  int party_;
  int port_;
  std::string address_;
  sci::IOPack *iopack_ = nullptr;
  sci::OTPack *otpack_ = nullptr;
  MathFunctions *math_ = nullptr;
};

PYBIND11_MODULE(litepi_sci, m) {
  m.doc() =
      "LitePI SCI/CrypTFlow2 MPC backend: constant-round ring ReLU on "
      "two-party additive shares (EzPC SCI, silent-OT/VOLE OT setup).";

  py::class_<SecureReLU>(m, "SecureReLU")
      .def(py::init<int, int, const std::string &>(),
           py::arg("party"), py::arg("port"), py::arg("address") = "127.0.0.1",
           "party=1 is ALICE (listens), party=2 is BOB (connects)")
      .def("relu", &SecureReLU::relu, py::arg("shares"), py::arg("bw"),
           "Secure ReLU on additive shares modulo 2^bw; returns each party's "
           "output share in the same ring.")
      .def("bytes_sent", &SecureReLU::bytes_sent,
           "Bytes exchanged on the SCI channels so far (incl. base OTs).")
      .def("rounds", &SecureReLU::rounds,
           "Round count reported by SCI's socket layer.");
}
