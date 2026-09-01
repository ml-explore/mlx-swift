// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import XCTest

class FFTTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    func testFFTRoundTrip() {
        let a = MLXArray(converting: [1.0, 2.0, 3.0, 4.0])

        for norm in [FFTNorm.backward, .ortho, .forward] {
            let r = ifft(fft(a, norm: norm), norm: norm)
            assertEqual(r.realPart(), a, atol: 1e-5)
        }
    }

    func testFFTNormScaling() {
        let a = MLXArray(converting: [1.0, 1.0, 1.0, 1.0])
        let n = 4.0

        // for a constant signal only the DC term is non-zero and its magnitude
        // shows how each mode scales the forward transform
        let backward = fft(a, norm: .backward)[0].realPart().item(Double.self)
        let ortho = fft(a, norm: .ortho)[0].realPart().item(Double.self)
        let forward = fft(a, norm: .forward)[0].realPart().item(Double.self)

        XCTAssertEqual(backward, n, accuracy: 1e-5)
        XCTAssertEqual(ortho, n / n.squareRoot(), accuracy: 1e-5)
        XCTAssertEqual(forward, 1.0, accuracy: 1e-5)
    }

    func testFFTNormDefaultIsBackward() {
        let a = MLXArray(converting: [1.0, 2.0, 3.0, 4.0])
        assertEqual(fft(a).realPart(), fft(a, norm: .backward).realPart(), atol: 1e-5)
    }

    func testFFTNAndAxes() {
        let a = MLXArray(0 ..< 24, [2, 3, 4]).asType(.float32)

        // norm should thread through each of the n-dimensional variants
        for norm in [FFTNorm.backward, .ortho, .forward] {
            assertEqual(ifftn(fftn(a, norm: norm), norm: norm).realPart(), a, atol: 1e-4)
            assertEqual(
                ifft2(fft2(a, norm: norm), norm: norm).realPart(), a, atol: 1e-4)
            assertEqual(
                irfftn(rfftn(a, norm: norm), norm: norm), a, atol: 1e-4)
        }
    }

    func testFFTFrequencies() {
        // matches numpy/mlx: [0, 1, 2, 3, -4, -3, -2, -1] / (n * d)
        let f = fftfreq(8)
        assertEqual(
            f, MLXArray(converting: [0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125]),
            atol: 1e-6)

        // d scales the result
        let scaled = fftfreq(8, d: 0.5)
        assertEqual(scaled, f * 2, atol: 1e-6)
    }

    func testRFFTFrequencies() {
        // only the non-negative frequencies -- n / 2 + 1 of them
        let f = rfftfreq(8)
        XCTAssertEqual(f.shape, [5])
        assertEqual(f, MLXArray(converting: [0, 0.125, 0.25, 0.375, 0.5]), atol: 1e-6)
    }
}
