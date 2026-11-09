// litepi_sci.cpp - pybind11 wrapper for SCI (CrypTFlow2) ReLU protocol.
//
// Exposes a two-party secure ReLU to Python. Each party (server/client)
// holds an additive share of the input and receives an additive share of
// the output. The protocol runs in constant rounds (r_h = 1) using
// CrypTFlow2's millionaire protocol over OT extension.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

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
  }

  ~SecureReLU() {
    delete math_;
    delete otpack_;
    delete iopack_;
  }

  // Secure ReLU over an array of fixed-point shares.
  // Input/output are uint64 arrays representing two's-complement shares.
  // bw: bit-width of the fixed-point representation.
  py::array_t<uint64_t> relu(py::array_t<uint64_t> shares, int bw) {
    py::buffer_info buf = shares.request();
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

  // Communication bytes sent since construction (for profiling).
  uint64_t bytes_sent() { return iopack_->get_comm(); }

private:
  int party_;
  int port_;
  std::string address_;
  IOPack *iopack_;
  OTPack *otpack_;
  MathFunctions *math_;
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
      .def("bytes_sent", &SecureReLU::bytes_sent);

  m.attr("ALICE") = 1;
  m.attr("BOB") = 2;
}
