// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// MLX-Swift Tutorial Program
///
/// This tutorial demonstrates MLX array operations, automatic differentiation,
/// and best practices for MLX-Swift development.
///
/// Based on: [MLX C++ Tutorial](https://github.com/ml-explore/mlx/blob/main/examples/cpp/tutorial.cpp)
@main
struct Tutorial {

    // MARK: - Constants

    private static let tolerance: Float = 1e-6
    private static let defaultArrayShape = [2, 2]
    private static let sampleValues: [Float] = [1.0, 2.0, 3.0, 4.0]

    // MARK: - Error Types

    enum TutorialError: Error, CustomStringConvertible {
        case invalidDataType(expected: DType, actual: DType)
        case invalidShape(expected: [Int], actual: [Int])
        case invalidValue(expected: Float, actual: Float)
        case arrayEvaluationFailed

        var description: String {
            switch self {
            case .invalidDataType(let expected, let actual):
                return "Invalid data type: expected \(expected), got \(actual)"
            case .invalidShape(let expected, let actual):
                return "Invalid shape: expected \(expected), got \(actual)"
            case .invalidValue(let expected, let actual):
                return "Invalid value: expected \(expected), got \(actual)"
            case .arrayEvaluationFailed:
                return "Array evaluation failed"
            }
        }
    }

    // MARK: - Utility Functions

    /// Validates that two floating-point values are approximately equal
    private static func isApproximatelyEqual(_ a: Float, _ b: Float, tolerance: Float = tolerance)
        -> Bool
    {
        abs(a - b) < tolerance
    }

    /// Safely validates array properties with proper error handling
    private static func validateArray(
        _ array: MLXArray,
        expectedDataType: DType? = nil,
        expectedShape: [Int]? = nil,
        expectedSize: Int? = nil,
        expectedDimensions: Int? = nil
    ) throws {
        // Validate data type
        if let expectedType = expectedDataType {
            guard array.dtype == expectedType else {
                throw TutorialError.invalidDataType(expected: expectedType, actual: array.dtype)
            }
        }

        // Validate shape
        if let expectedShape = expectedShape {
            guard array.shape == expectedShape else {
                throw TutorialError.invalidShape(expected: expectedShape, actual: array.shape)
            }
        }

        // Validate size
        if let expectedSize = expectedSize {
            guard array.size == expectedSize else {
                throw TutorialError.invalidValue(
                    expected: Float(expectedSize), actual: Float(array.size))
            }
        }

        // Validate dimensions
        if let expectedDimensions = expectedDimensions {
            guard array.ndim == expectedDimensions else {
                throw TutorialError.invalidValue(
                    expected: Float(expectedDimensions), actual: Float(array.ndim))
            }
        }
    }

    /// Demonstrates safe item extraction with error handling
    private static func safeExtractValue<T: HasDType>(_ array: MLXArray, as type: T.Type) throws
        -> T
    {
        // Ensure array is evaluated before extraction
        array.eval()

        guard array.size == 1 else {
            throw TutorialError.invalidValue(expected: 1, actual: Float(array.size))
        }

        return array.item(type)
    }

    // MARK: - Tutorial Sections

    /// Demonstrates scalar array operations with comprehensive validation
    static func demonstrateScalarOperations() throws {
        print("=== Scalar Operations Demo ===")

        // Create a scalar array with explicit type
        let scalarValue: Float = 1.0
        let scalarArray = MLXArray(scalarValue)

        // Validate scalar properties
        try validateArray(
            scalarArray,
            expectedDataType: .float32,
            expectedShape: [],
            expectedSize: 1,
            expectedDimensions: 0
        )

        // Safe value extraction
        let extractedValue = try safeExtractValue(scalarArray, as: Float.self)
        guard isApproximatelyEqual(extractedValue, scalarValue) else {
            throw TutorialError.invalidValue(expected: scalarValue, actual: extractedValue)
        }

        print("✓ Scalar array created: \(scalarArray)")
        print("✓ Data type: \(scalarArray.dtype)")
        print("✓ Extracted value: \(extractedValue)")
        print("✓ Shape: \(scalarArray.shape)")
        print("✓ Size: \(scalarArray.size)")
        print("✓ Dimensions: \(scalarArray.ndim)")

        // Demonstrate type safety
        demonstrateTypeSafety(with: scalarArray)

        print()
    }

    /// Demonstrates type safety and error handling
    private static func demonstrateTypeSafety(with array: MLXArray) {
        print("--- Type Safety Demo ---")

        // This would cause a runtime error in the original code
        // We demonstrate safe handling instead
        do {
            guard array.dtype == .int32 else {
                print("⚠️  Cannot safely extract as Int32 from \(array.dtype) array")
                return
            }
            let _ = try safeExtractValue(array, as: Int32.self)
        } catch {
            print("⚠️  Type safety check passed: \(error)")
        }
    }

    /// Demonstrates multidimensional array operations with performance optimizations
    static func demonstrateArrayOperations() throws {
        print("=== Array Operations Demo ===")

        // Create multidimensional array with better performance
        let sourceData = sampleValues
        let arrayShape = defaultArrayShape
        let multiArray = MLXArray(sourceData, arrayShape)

        // Validate array properties
        try validateArray(
            multiArray,
            expectedDataType: .float32,
            expectedShape: arrayShape
        )

        print("✓ Multidimensional array created: \n\(multiArray)")
        print("✓ Shape: \(multiArray.shape)")

        // Demonstrate row access with bounds checking
        try demonstrateRowAccess(multiArray)

        // Create ones array for operations
        let onesArray = MLXArray.ones(arrayShape)
        try validateArray(onesArray, expectedShape: arrayShape)

        // Perform lazy computation
        let computationResult = try performLazyComputation(multiArray, onesArray)

        // Demonstrate evaluation strategies
        try demonstrateEvaluationStrategies(computationResult)

        print()
    }

    /// Demonstrates safe row access with bounds checking
    private static func demonstrateRowAccess(_ array: MLXArray) throws {
        print("--- Row Access Demo ---")

        guard array.ndim >= 2 && array.shape[0] >= 2 else {
            throw TutorialError.invalidShape(expected: [2, 2], actual: array.shape)
        }

        // Safe row access with validation
        let firstRow = array[0]
        let secondRow = array[1]

        print("✓ First row: \(firstRow)")
        print("✓ Second row: \(secondRow)")

        // Validate row shapes
        let expectedRowShape = Array(array.shape.dropFirst())
        try validateArray(firstRow, expectedShape: expectedRowShape)
        try validateArray(secondRow, expectedShape: expectedRowShape)
    }

    /// Performs lazy computation with proper validation
    private static func performLazyComputation(_ x: MLXArray, _ y: MLXArray) throws -> MLXArray {
        print("--- Lazy Computation Demo ---")

        // Ensure arrays are compatible for operation
        guard x.shape == y.shape else {
            throw TutorialError.invalidShape(expected: x.shape, actual: y.shape)
        }

        // Perform pointwise addition (lazy)
        let result = x + y

        // Validate lazy computation properties
        try validateArray(
            result,
            expectedDataType: .float32,
            expectedShape: x.shape
        )

        print("✓ Lazy computation created")
        print("✓ Result shape: \(result.shape)")
        print("✓ Result data type: \(result.dtype)")

        return result
    }

    /// Demonstrates different evaluation strategies
    private static func demonstrateEvaluationStrategies(_ array: MLXArray) throws {
        print("--- Evaluation Strategies Demo ---")

        // Measure evaluation time
        let startTime = CFAbsoluteTimeGetCurrent()

        // Explicit evaluation
        array.eval()

        let evaluationTime = CFAbsoluteTimeGetCurrent() - startTime
        print("✓ Explicit evaluation completed in \(String(format: "%.4f", evaluationTime))s")

        // Implicit evaluation through printing
        print("✓ Array after evaluation: \n\(array)")

        // Validate that array now has data
        guard array.size > 0 else {
            throw TutorialError.arrayEvaluationFailed
        }
    }

    /// Demonstrates automatic differentiation with enhanced error handling
    static func demonstrateAutomaticDifferentiation() throws {
        print("=== Automatic Differentiation Demo ===")

        // Define a more complex function for differentiation
        func squareFunction(_ x: MLXArray) -> MLXArray {
            return x.square()
        }

        func complexFunction(_ x: MLXArray) -> MLXArray {
            return x.square() + 2 * x + 1  // f(x) = x² + 2x + 1
        }

        // Test input value
        let testValue: Float = 1.5
        let inputArray = MLXArray(testValue)

        // Demonstrate first derivative
        try demonstrateFirstDerivative(
            function: squareFunction, input: inputArray, testValue: testValue)

        // Demonstrate second derivative
        try demonstrateSecondDerivative(
            function: squareFunction, input: inputArray, testValue: testValue)

        // Demonstrate complex function differentiation
        try demonstrateComplexDifferentiation(
            function: complexFunction, input: inputArray, testValue: testValue)

        print()
    }

    /// Demonstrates first derivative computation
    private static func demonstrateFirstDerivative(
        function: @escaping (MLXArray) -> MLXArray,
        input: MLXArray,
        testValue: Float
    ) throws {
        print("--- First Derivative Demo ---")

        let gradientFunction = grad(function)
        let firstDerivative = gradientFunction(input)

        let derivativeValue = try safeExtractValue(firstDerivative, as: Float.self)
        let expectedValue: Float = 2 * testValue  // d/dx(x²) = 2x

        guard isApproximatelyEqual(derivativeValue, expectedValue) else {
            throw TutorialError.invalidValue(expected: expectedValue, actual: derivativeValue)
        }

        print("✓ First derivative: \(firstDerivative)")
        print("✓ Expected: \(expectedValue), Got: \(derivativeValue)")
    }

    /// Demonstrates second derivative computation
    private static func demonstrateSecondDerivative(
        function: @escaping (MLXArray) -> MLXArray,
        input: MLXArray,
        testValue: Float
    ) throws {
        print("--- Second Derivative Demo ---")

        let secondGradientFunction = grad(grad(function))
        let secondDerivative = secondGradientFunction(input)

        let secondDerivativeValue = try safeExtractValue(secondDerivative, as: Float.self)
        let expectedValue: Float = 2.0  // d²/dx²(x²) = 2

        guard isApproximatelyEqual(secondDerivativeValue, expectedValue) else {
            throw TutorialError.invalidValue(expected: expectedValue, actual: secondDerivativeValue)
        }

        print("✓ Second derivative: \(secondDerivative)")
        print("✓ Expected: \(expectedValue), Got: \(secondDerivativeValue)")
    }

    /// Demonstrates differentiation of complex functions
    private static func demonstrateComplexDifferentiation(
        function: @escaping (MLXArray) -> MLXArray,
        input: MLXArray,
        testValue: Float
    ) throws {
        print("--- Complex Function Differentiation Demo ---")

        let gradientFunction = grad(function)
        let derivative = gradientFunction(input)

        let derivativeValue = try safeExtractValue(derivative, as: Float.self)
        let expectedValue: Float = 2 * testValue + 2  // d/dx(x² + 2x + 1) = 2x + 2

        guard isApproximatelyEqual(derivativeValue, expectedValue) else {
            throw TutorialError.invalidValue(expected: expectedValue, actual: derivativeValue)
        }

        print("✓ Complex function derivative: \(derivative)")
        print("✓ Expected: \(expectedValue), Got: \(derivativeValue)")
    }

    /// Demonstrates performance benchmarking
    private static func demonstratePerformanceBenchmark() {
        print("=== Performance Benchmark Demo ===")

        let iterations = 1000
        let arraySize = [100, 100]

        // Benchmark array creation
        let creationTime = measureExecutionTime {
            for _ in 0 ..< iterations {
                let _ = MLXArray.ones(arraySize)
            }
        }

        print(
            "✓ Array creation (\(iterations) iterations): \(String(format: "%.4f", creationTime))s")

        // Benchmark array operations
        let x = MLXArray.ones(arraySize)
        let y = MLXArray.ones(arraySize)

        let operationTime = measureExecutionTime {
            for _ in 0 ..< iterations {
                let _ = x + y
            }
        }

        print(
            "✓ Array operations (\(iterations) iterations): \(String(format: "%.4f", operationTime))s"
        )
        print()
    }

    /// Measures execution time of a given operation
    private static func measureExecutionTime(_ operation: () -> Void) -> Double {
        let startTime = CFAbsoluteTimeGetCurrent()
        operation()
        return CFAbsoluteTimeGetCurrent() - startTime
    }

    /// Enhanced main function with comprehensive error handling
    static func main() {
        print("🚀 Enhanced MLX-Swift Tutorial\n")

        do {
            try demonstrateScalarOperations()
            try demonstrateArrayOperations()
            try demonstrateAutomaticDifferentiation()
            demonstratePerformanceBenchmark()
            try demonstrateAdvancedFeatures()

            print("✅ All tutorial sections completed successfully!")

        } catch let error as TutorialError {
            print("❌ Tutorial error: \(error)")
            exit(1)
        } catch {
            print("❌ Unexpected error: \(error)")
            exit(1)
        }
    }
}

// MARK: - Extensions

extension Tutorial {
    /// Demonstrates additional MLX features and patterns
    static func demonstrateAdvancedFeatures() throws {
        print("=== Advanced Features Demo ===")

        // Demonstrate different data types
        try demonstrateDataTypes()

        // Demonstrate broadcasting
        try demonstrateBroadcasting()

        // Demonstrate memory management
        demonstrateMemoryManagement()

        print()
    }

    private static func demonstrateDataTypes() throws {
        print("--- Data Types Demo ---")

        let int32Array = MLXArray([1, 2, 3, 4]).asType(.int32)
        let float32Array = MLXArray(converting: [1.0, 2.0, 3.0, 4.0])
        let boolArray = MLXArray([true, false, true, false])

        try validateArray(int32Array, expectedDataType: .int32)
        try validateArray(float32Array, expectedDataType: .float32)
        try validateArray(boolArray, expectedDataType: .bool)

        print("✓ Int32 array: \(int32Array)")
        print("✓ Float32 array: \(float32Array)")
        print("✓ Bool array: \(boolArray)")
    }

    private static func demonstrateBroadcasting() throws {
        print("--- Broadcasting Demo ---")

        let matrix = MLXArray.ones([3, 4])
        let vector = MLXArray(converting: [1.0, 2.0, 3.0, 4.0])

        // Broadcasting addition
        let result = matrix + vector

        try validateArray(result, expectedShape: [3, 4])
        print("✓ Broadcasting result shape: \(result.shape)")
        print("✓ Broadcasting result: \n\(result)")
    }

    private static func demonstrateMemoryManagement() {
        print("--- Memory Management Demo ---")

        autoreleasepool {
            let largeArray = MLXArray.zeros([1000, 1000])
            largeArray.eval()
            print("✓ Large array created and evaluated")
        }

        print("✓ Memory released after autoreleasepool")
    }
}
