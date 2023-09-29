// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Values in Montgomery form

const fr_t group_gen = fr_t(0x2ffffffa);
const fr_t group_gen_inverse = fr_t(0x2d555555);

const int S = 27;

const fr_t forward_roots_of_unity[S + 1] = {
    fr_t(0x0ffffffe),
    fr_t(0x68000003),
    fr_t(0x5bc72af0),
    fr_t(0x02ec07f3),
    fr_t(0x67e027ca),
    fr_t(0x19e5f901),
    fr_t(0x3b27e54a),
    fr_t(0x20d1773e),
    fr_t(0x771ea53a),
    fr_t(0x0fb182ad),
    fr_t(0x146d1455),
    fr_t(0x3e7d65f0),
    fr_t(0x327884f2),
    fr_t(0x53fc8703),
    fr_t(0x20742dd1),
    fr_t(0x31062eda),
    fr_t(0x642b70ab),
    fr_t(0x1ccd534b),
    fr_t(0x03cc9bf7),
    fr_t(0x6686182f),
    fr_t(0x2e2516d3),
    fr_t(0x5701b5c8),
    fr_t(0x193a6352),
    fr_t(0x112fc5b9),
    fr_t(0x63ec6b91),
    fr_t(0x5b34b3ff),
    fr_t(0x3fff6398),
    fr_t(0x1ffffedc)
};

const fr_t inverse_roots_of_unity[S + 1] = {
    fr_t(0x0ffffffe),
    fr_t(0x68000003),
    fr_t(0x1c38d511),
    fr_t(0x3d85298f),
    fr_t(0x5f06e481),
    fr_t(0x38a3c615),
    fr_t(0x4ed6e525),
    fr_t(0x55372b64),
    fr_t(0x4d88ae94),
    fr_t(0x5806fd5e),
    fr_t(0x2ced6d6a),
    fr_t(0x1851eacd),
    fr_t(0x2fa36b4d),
    fr_t(0x0a556a3b),
    fr_t(0x18ae7209),
    fr_t(0x742ba568),
    fr_t(0x3f462cba),
    fr_t(0x50b5c3b2),
    fr_t(0x0dfdfca6),
    fr_t(0x3821b546),
    fr_t(0x45e4cd80),
    fr_t(0x3e6793bd),
    fr_t(0x5bdeafa3),
    fr_t(0x2e01d37a),
    fr_t(0x2da9f4f0),
    fr_t(0x1db7e183),
    fr_t(0x167ca34b),
    fr_t(0x50b3630a)
};

const fr_t domain_size_inverse[S + 1] = {
    fr_t(0x0ffffffeu),
    fr_t(0x07ffffffu),
    fr_t(0x40000000u),
    fr_t(0x20000000u),
    fr_t(0x10000000u),
    fr_t(0x08000000u),
    fr_t(0x04000000u),
    fr_t(0x02000000u),
    fr_t(0x01000000u),
    fr_t(0x00800000u),
    fr_t(0x00400000u),
    fr_t(0x00200000u),
    fr_t(0x00100000u),
    fr_t(0x00080000u),
    fr_t(0x00040000u),
    fr_t(0x00020000u),
    fr_t(0x00010000u),
    fr_t(0x00008000u),
    fr_t(0x00004000u),
    fr_t(0x00002000u),
    fr_t(0x00001000u),
    fr_t(0x00000800u),
    fr_t(0x00000400u),
    fr_t(0x00000200u),
    fr_t(0x00000100u),
    fr_t(0x00000080u),
    fr_t(0x00000040u),
    fr_t(0x00000020u)
};
