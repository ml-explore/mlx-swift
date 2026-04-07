// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

// MARK: - Protocole de conversion de tableaux imbriqués

/// Protocole permettant de convertir un tableau Swift imbriqué en MLXArray.
///
/// Les types scalaires conformes à ``HasDType`` servent de feuilles,
/// et `Array<Element: MLXNestedArray>` se conforme récursivement pour
/// gérer n'importe quelle profondeur d'imbrication.
///
/// ### Exemple
/// ```swift
/// let matrix = MLXArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
/// // shape: [2, 3], dtype: .float32
///
/// let cube = MLXArray([[[1, 0], [0, 1]], [[2, 0], [0, 2]]])
/// // shape: [2, 2, 2], dtype: .int32
/// ```
public protocol MLXNestedArray {
    /// Type scalaire des feuilles du tableau imbriqué.
    associatedtype ScalarType: HasDType

    /// Forme (shape) du tableau imbriqué à ce niveau.
    var mlxShape: [Int] { get }

    /// Valeurs aplaties dans l'ordre row-major (C-order).
    func mlxFlattenedValues() -> [ScalarType]
}

// MARK: - Conformance récursive de Array

extension Array: MLXNestedArray where Element: MLXNestedArray {
    public typealias ScalarType = Element.ScalarType

    /// Calcule la forme en ajoutant la dimension actuelle devant la shape de l'élément.
    ///
    /// Précondition : tous les éléments ont la même shape (tableau rectangulaire).
    public var mlxShape: [Int] {
        guard let first = first else {
            // tableau vide — on retourne [0] suivi de la shape interne de zéro
            return [0]
        }
        return [count] + first.mlxShape
    }

    /// Aplatit récursivement tous les éléments en un tableau 1D de scalaires.
    public func mlxFlattenedValues() -> [ScalarType] {
        flatMap { $0.mlxFlattenedValues() }
    }
}

// MARK: - Conformances scalaires

extension Bool: MLXNestedArray {
    public typealias ScalarType = Bool
    public var mlxShape: [Int] { [] }
    public func mlxFlattenedValues() -> [Bool] { [self] }
}

extension Int32: MLXNestedArray {
    public typealias ScalarType = Int32
    public var mlxShape: [Int] { [] }
    public func mlxFlattenedValues() -> [Int32] { [self] }
}

extension Int64: MLXNestedArray {
    public typealias ScalarType = Int64
    public var mlxShape: [Int] { [] }
    public func mlxFlattenedValues() -> [Int64] { [self] }
}

extension UInt8: MLXNestedArray {
    public typealias ScalarType = UInt8
    public var mlxShape: [Int] { [] }
    public func mlxFlattenedValues() -> [UInt8] { [self] }
}

extension UInt16: MLXNestedArray {
    public typealias ScalarType = UInt16
    public var mlxShape: [Int] { [] }
    public func mlxFlattenedValues() -> [UInt16] { [self] }
}

extension UInt32: MLXNestedArray {
    public typealias ScalarType = UInt32
    public var mlxShape: [Int] { [] }
    public func mlxFlattenedValues() -> [UInt32] { [self] }
}

extension Float32: MLXNestedArray {
    public typealias ScalarType = Float32
    public var mlxShape: [Int] { [] }
    public func mlxFlattenedValues() -> [Float32] { [self] }
}

extension Float64: MLXNestedArray {
    public typealias ScalarType = Float64
    public var mlxShape: [Int] { [] }
    public func mlxFlattenedValues() -> [Float64] { [self] }
}

#if !arch(x86_64)
    extension Float16: MLXNestedArray {
        public typealias ScalarType = Float16
        public var mlxShape: [Int] { [] }
        public func mlxFlattenedValues() -> [Float16] { [self] }
    }
#endif

// MARK: - Initialiseur MLXArray pour tableaux imbriqués

extension MLXArray {

    /// Crée un ``MLXArray`` multi-dimensionnel à partir d'un tableau Swift imbriqué.
    ///
    /// Reproduit le comportement ergonomique de `mx.array([[1, 2], [3, 4]])` en Python.
    /// La shape est déduite automatiquement de la structure d'imbrication.
    ///
    /// ```swift
    /// // Tableau 2D : shape [2, 3], dtype .float32
    /// let matrix = MLXArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    ///
    /// // Tableau 3D : shape [2, 2, 2], dtype .int32
    /// let cube = MLXArray([[[1, 0], [0, 1]], [[2, 0], [0, 2]]])
    /// ```
    ///
    /// - Note: Le tableau doit être rectangulaire (tous les sous-tableaux à chaque
    ///   profondeur ont la même taille). La conformité est vérifiée par precondition.
    ///
    /// - Parameter nested: Tableau Swift imbriqué dont les feuilles sont des scalaires
    ///   conformes à ``HasDType``.
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init<N: MLXNestedArray>(_ nested: N) where N: Collection, N.Element: MLXNestedArray {
        // Validation : tous les sous-tableaux à ce niveau doivent avoir la même shape
        let shape = nested.mlxShape
        let flatValues = nested.mlxFlattenedValues()

        // Vérifie la cohérence de la shape avec le nombre d'éléments aplatis
        let expectedCount = shape.isEmpty ? 1 : shape.reduce(1, *)
        precondition(
            flatValues.count == expectedCount,
            "Tableau imbriqué irrégulier : shape \(shape) attend \(expectedCount) éléments, \(flatValues.count) trouvés. "
                + "Vérifiez que tous les sous-tableaux ont la même longueur."
        )

        self.init(flatValues, shape)
    }

    /// Crée un ``MLXArray`` 2D à partir d'un tableau de tableaux.
    ///
    /// Surcharge dédiée aux tableaux 2D (le cas d'usage le plus fréquent),
    /// offrant une meilleure inférence de type au call-site.
    ///
    /// ```swift
    /// let matrix = MLXArray([[1, 2, 3], [4, 5, 6]])
    /// // shape: [2, 3], dtype: .int32
    ///
    /// let floatMatrix = MLXArray([[0.5, 1.5], [2.5, 3.5]])
    /// // shape: [2, 2], dtype: .float32
    /// ```
    ///
    /// - Parameter rows: Tableau 2D — chaque élément est une ligne.
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``init(_:)-([[MLXNestedArray]])``
    public convenience init<T: HasDType>(_ rows: [[T]]) {
        let rowCount = rows.count

        guard rowCount > 0 else {
            // tableau vide : shape [0]
            self.init([T](), [0])
            return
        }

        let colCount = rows[0].count

        // Vérifie que toutes les lignes ont la même largeur
        precondition(
            rows.allSatisfy { $0.count == colCount },
            "Tableau 2D irrégulier : toutes les lignes doivent avoir la même longueur (\(colCount) éléments attendus)."
        )

        let flat = rows.flatMap { $0 }
        self.init(flat, [rowCount, colCount])
    }

    /// Crée un ``MLXArray`` 2D à partir d'un tableau de tableaux d'`Int`.
    ///
    /// Produit un tableau de dtype `.int32` (le comportement par défaut pour `Int` dans MLX Swift).
    ///
    /// ```swift
    /// let a = MLXArray([[7, 8], [9, 10]])
    /// // shape: [2, 2], dtype: .int32
    /// ```
    ///
    /// - Parameter rows: Tableau 2D d'entiers.
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init(_ rows: [[Int]]) {
        let rowCount = rows.count

        guard rowCount > 0 else {
            self.init([Int32](), [0])
            return
        }

        let colCount = rows[0].count

        precondition(
            rows.allSatisfy { $0.count == colCount },
            "Tableau 2D irrégulier : toutes les lignes doivent avoir la même longueur (\(colCount) éléments attendus)."
        )

        precondition(
            rows.joined().allSatisfy { (Int(Int32.min)...Int(Int32.max)).contains($0) },
            "Valeur hors limites pour Int32 — utilisez [[Int32]] si les valeurs dépassent Int32.max."
        )

        let flat = rows.flatMap { $0 }.map { Int32($0) }
        self.init(flat, [rowCount, colCount])
    }

    /// Crée un ``MLXArray`` 3D à partir d'un tableau de tableaux de tableaux.
    ///
    /// ```swift
    /// let cube = MLXArray([[[1, 0], [0, 1]], [[2, 0], [0, 2]]])
    /// // shape: [2, 2, 2], dtype: .int32
    /// ```
    ///
    /// - Parameter slices: Tableau 3D — chaque élément est une matrice 2D.
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init<T: HasDType>(_ slices: [[[T]]]) {
        let depth = slices.count

        guard depth > 0 else {
            self.init([T](), [0])
            return
        }

        let rowCount = slices[0].count
        let colCount = slices[0].first?.count ?? 0

        precondition(
            slices.allSatisfy { $0.count == rowCount },
            "Tableau 3D irrégulier : toutes les tranches doivent avoir \(rowCount) lignes."
        )
        precondition(
            slices.allSatisfy { $0.allSatisfy { $0.count == colCount } },
            "Tableau 3D irrégulier : toutes les lignes doivent avoir \(colCount) colonnes."
        )

        let flat = slices.flatMap { $0.flatMap { $0 } }
        self.init(flat, [depth, rowCount, colCount])
    }

    /// Crée un ``MLXArray`` 3D à partir d'un tableau de tableaux de tableaux d'`Int`.
    ///
    /// Produit un tableau de dtype `.int32`.
    ///
    /// ```swift
    /// let cube = MLXArray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    /// // shape: [2, 2, 2], dtype: .int32
    /// ```
    ///
    /// - Parameter slices: Tableau 3D d'entiers.
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init(_ slices: [[[Int]]]) {
        let depth = slices.count

        guard depth > 0 else {
            self.init([Int32](), [0])
            return
        }

        let rowCount = slices[0].count
        let colCount = slices[0].first?.count ?? 0

        precondition(
            slices.allSatisfy { $0.count == rowCount },
            "Tableau 3D irrégulier : toutes les tranches doivent avoir \(rowCount) lignes."
        )
        precondition(
            slices.allSatisfy { $0.allSatisfy { $0.count == colCount } },
            "Tableau 3D irrégulier : toutes les lignes doivent avoir \(colCount) colonnes."
        )

        let allValues = slices.joined().joined()
        precondition(
            allValues.allSatisfy { (Int(Int32.min)...Int(Int32.max)).contains($0) },
            "Valeur hors limites pour Int32 — utilisez [[[Int32]]] si les valeurs dépassent Int32.max."
        )

        let flat = slices.flatMap { $0.flatMap { $0 } }.map { Int32($0) }
        self.init(flat, [depth, rowCount, colCount])
    }
}
