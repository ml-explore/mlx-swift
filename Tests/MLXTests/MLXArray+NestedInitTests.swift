// Copyright © 2024 Apple Inc.

import Foundation
import XCTest

@testable import MLX

// Tests pour les initialiseurs de tableaux imbriqués (issue #161)
class MLXArrayNestedInitTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    // MARK: - Tableaux 2D avec types génériques

    func testInit2DFloat() {
        // Tableau 2D de Float32
        let matrix = MLXArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] as [[Float32]])
        XCTAssertEqual(matrix.shape, [2, 3])
        XCTAssertEqual(matrix.dtype, .float32)
        XCTAssertEqual(matrix.size, 6)
    }

    func testInit2DInt32() {
        // Tableau 2D de Int32
        let matrix = MLXArray([[1, 2], [3, 4]] as [[Int32]])
        XCTAssertEqual(matrix.shape, [2, 2])
        XCTAssertEqual(matrix.dtype, .int32)
        XCTAssertEqual(matrix.size, 4)
    }

    func testInit2DInt() {
        // Tableau 2D d'Int (surcharge dédiée, produit du .int32)
        let matrix = MLXArray([[7, 8], [9, 10]])
        XCTAssertEqual(matrix.shape, [2, 2])
        XCTAssertEqual(matrix.dtype, .int32)
        XCTAssertEqual(matrix.size, 4)
    }

    func testInit2DIntValues() {
        // Vérifie que les valeurs sont correctement stockées
        let matrix = MLXArray([[1, 2], [3, 4]])
        let expected = MLXArray([1, 2, 3, 4] as [Int32], [2, 2])
        assertEqual(matrix, expected)
    }

    func testInit2DFloatValues() {
        // Vérifie que les valeurs float sont correctement stockées
        let matrix = MLXArray([[1.0, 2.0], [3.0, 4.0]] as [[Float32]])
        let expected = MLXArray([1.0, 2.0, 3.0, 4.0] as [Float32], [2, 2])
        assertEqual(matrix, expected)
    }

    func testInit2DNonSquare() {
        // Matrice non carrée
        let matrix = MLXArray([[1, 2, 3], [4, 5, 6]] as [[Int32]])
        XCTAssertEqual(matrix.shape, [2, 3])
        XCTAssertEqual(matrix.ndim, 2)
        XCTAssertEqual(matrix.dim(0), 2)
        XCTAssertEqual(matrix.dim(1), 3)
    }

    func testInit2DFloat64() {
        // Tableau 2D de Double (Float64)
        let matrix = MLXArray([[1.5, 2.5], [3.5, 4.5]] as [[Double]])
        XCTAssertEqual(matrix.shape, [2, 2])
        XCTAssertEqual(matrix.dtype, .float64)
    }

    func testInit2DBool() {
        // Tableau 2D de Bool
        let matrix = MLXArray([[true, false], [false, true]])
        XCTAssertEqual(matrix.shape, [2, 2])
        XCTAssertEqual(matrix.dtype, .bool)
    }

    // MARK: - Tableaux 3D

    func testInit3DInt() {
        // Cube 3D d'Int (produit du .int32)
        let cube = MLXArray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        XCTAssertEqual(cube.shape, [2, 2, 2])
        XCTAssertEqual(cube.dtype, .int32)
        XCTAssertEqual(cube.size, 8)
    }

    func testInit3DFloat() {
        // Cube 3D de Float32
        let cube = MLXArray([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]] as [[[Float32]]])
        XCTAssertEqual(cube.shape, [2, 2, 2])
        XCTAssertEqual(cube.dtype, .float32)
        XCTAssertEqual(cube.size, 8)
    }

    func testInit3DIntValues() {
        // Vérifie les valeurs pour le cas 3D
        let cube = MLXArray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        let expected = MLXArray([1, 2, 3, 4, 5, 6, 7, 8] as [Int32], [2, 2, 2])
        assertEqual(cube, expected)
    }

    func testInit3DNonCubic() {
        // Forme 3D non cubique : [2, 3, 4]
        let tensor = MLXArray([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ])
        XCTAssertEqual(tensor.shape, [2, 3, 4])
        XCTAssertEqual(tensor.ndim, 3)
        XCTAssertEqual(tensor.size, 24)
    }

    // MARK: - Tableaux vides

    func testInit2DEmptyRows() {
        // Tableau avec 0 lignes
        let empty = MLXArray([[Int32]]())
        XCTAssertEqual(empty.shape, [0])
    }

    func testInit3DEmpty() {
        // Tableau 3D vide
        let empty = MLXArray([[[Int32]]]())
        XCTAssertEqual(empty.shape, [0])
    }

    // MARK: - Compatibilité avec l'API existante

    func testNestedInitCompatibleWithExisting1D() {
        // S'assure que la surcharge 2D n'interfère pas avec le init 1D existant
        let oneDim = MLXArray([1, 2, 3, 4])
        XCTAssertEqual(oneDim.shape, [4])
        XCTAssertEqual(oneDim.ndim, 1)
    }

    func testNestedInitIndexing() {
        // Vérifie l'accès par index après construction imbriquée
        let matrix = MLXArray([[10, 20], [30, 40]])
        let row0 = matrix[0]
        XCTAssertEqual(row0.shape, [2])
        let val = matrix[0, 1].item(Int32.self)
        XCTAssertEqual(val, 20)
    }
}
