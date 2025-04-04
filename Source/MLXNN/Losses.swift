// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// Different types of loss reductions
public enum LossReduction: String, Sendable {
    /// take the `mean` of the loss. This produces a a scalar array
    case mean
    /// take the `sum` of the loss.  This produces a a scalar array.
    case sum
    /// take the loss as-is.  This produces an array the same shape as the input.
    case none

    public func reduce(loss: MLXArray) -> MLXArray {
        switch self {
        case .mean:
            MLX.mean(loss)
        case .sum:
            MLX.sum(loss)
        case .none:
            loss
        }
    }
}

/// Computes the cross entropy loss.
///
/// - Parameters:
///   - logits: unnormalized predicted logits
///   - targets: target values, as class indices or class probabilities
///   - weights: weights for each target
///   - axis: axis over which to compute softmax
///   - labelSmoothing: label smoothing factor, range [0, 1)
///   - reduction: reduction type
/// - Returns: computed cross entropy loss
///
/// ### See Also
/// - <doc:losses>
public func crossEntropy(
    logits: MLXArray, targets: MLXArray, weights: MLXArray? = nil, axis: Int = -1,
    labelSmoothing: Float = 0, reduction: LossReduction = .none
) -> MLXArray {
    guard (0.0 ..< 1.0).contains(labelSmoothing) else {
        fatalError("labelSmoothing must be in [0, 1): \(labelSmoothing)")
    }

    let targets_as_probs = targets.ndim == logits.ndim

    var score: MLXArray
    if targets_as_probs {
        score = sum(logits * targets, axis: axis)
    } else {
        score = takeAlong(logits, targets.expandedDimensions(axis: -1), axis: axis).squeezed(
            axis: -1)
    }

    let logSumExpLogits = logSumExp(logits, axis: axis)

    var loss: MLXArray
    if labelSmoothing > 0 {
        // adjust the true class score with label smoothing
        let adjustedScore = (1 - labelSmoothing) * score

        // calculate the mean logit across the classes for smoothed loss
        let meanLogits = logits.mean(axis: axis)
        let smoothedLoss = -meanLogits * labelSmoothing

        // combine the adjusted score and smoothed loss with the logsumexp logits
        loss = logSumExpLogits - adjustedScore + smoothedLoss
    } else {
        loss = logSumExpLogits - score
    }

    if let weights {
        precondition(weights.shape == targets.shape)
        loss = loss * weights
    }

    return reduction.reduce(loss: loss)
}

/// Computes the binary cross entropy loss.
///
/// By default, this function takes the pre-sigmoid logits, which results in a faster
/// and more precise loss. For improved numerical stability when `withLogits` is true,
/// the loss calculation clips the input probabilities (in log-space) to a minimum value
/// of `-100`.
///
/// - Parameters:
///   - logits: unnormalized predicted logits
///   - targets: binary target values in {0, 1}
///   - weights: optional weights for each target
///   - withLogits: whether the `logits` parameter is logits or probabilities
///   - reduction: reduction type
/// - Returns: computed binary cross entropy loss
///
/// ### See Also
/// - <doc:losses>
public func binaryCrossEntropy(
    logits: MLXArray, targets: MLXArray,
    weights: MLXArray? = nil, withLogits: Bool = true,
    reduction: LossReduction = .mean
) -> MLXArray {
    var loss: MLXArray
    if withLogits {
        loss = logAddExp(0, logits) - targets * logits
    } else {
        let logInputsClip = clip(log(logits), min: -100)
        let logInputsInverseClip = clip(log(1 - logits), min: -100)
        loss = -(targets * logInputsClip + (1 - targets) * logInputsInverseClip)
    }
    if let weights {
        precondition(weights.shape == loss.shape)
        loss *= weights
    }
    return reduction.reduce(loss: loss)
}

/// Computes the L1 loss
/// - Parameters:
///   - predictions: the predicted values
///   - targets: the target values
///   - reduction: reduction type
/// - Returns: computed L1 loss
///
/// ### See Also
/// - <doc:losses>
public func l1Loss(predictions: MLXArray, targets: MLXArray, reduction: LossReduction = .mean)
    -> MLXArray
{
    precondition(predictions.shape == targets.shape)
    let loss = abs(predictions - targets)
    return reduction.reduce(loss: loss)
}

/// Computes the mean squared error loss.
/// - Parameters:
///   - predictions: the predicted values
///   - targets: the target values
///   - reduction: reduction type
/// - Returns: computed mean squared error loss
///
/// ### See Also
/// - <doc:losses>
public func mseLoss(predictions: MLXArray, targets: MLXArray, reduction: LossReduction = .mean)
    -> MLXArray
{
    precondition(predictions.shape == targets.shape)
    let loss = square(predictions - targets)
    return reduction.reduce(loss: loss)
}

/// Computes the negative log likelihood loss.
/// - Parameters:
///   - inputs: predicted distribution in log space
///   - targets: the target values
///   - axis: distribution axis
///   - reduction: reduction type
/// - Returns: computed NLL loss
///
/// ### See Also
/// - <doc:losses>
public func nllLoss(
    inputs: MLXArray, targets: MLXArray, axis: Int = -1, reduction: LossReduction = .none
) -> MLXArray {
    let loss = -takeAlong(inputs, targets.expandedDimensions(axis: -1), axis: axis).squeezed(
        axis: -1)
    return reduction.reduce(loss: loss)
}

/// Computes the Kullback-Leibler divergence loss.
///
/// Computes the following when the `reduction: .none`:
///
/// ```swift
/// sum(exp(targets) * (targets - inputs), axis: axis)
/// ```
///
/// - Parameters:
///   - inputs: Log probabilities for the predicted distribution
///   - targets: Log probabilities for the target distribution
///   - axis: distribution axis
///   - reduction: reduction type
/// - Returns: computed Kullback-Leibler divergence loss
///
/// ### See Also
/// - <doc:losses>
public func klDivLoss(
    inputs: MLXArray, targets: MLXArray, axis: Int = -1, reduction: LossReduction = .none
) -> MLXArray {
    let loss = sum(exp(targets) * (targets - inputs), axis: axis)
    return reduction.reduce(loss: loss)
}

/// Computes the smooth L1 loss.
///
/// The smooth L1 loss is a variant of the L1 loss which replaces the absolute
/// difference with a squared difference when the absolute difference is less
/// than `beta`.
///
/// - Parameters:
///   - predictions: predicted values
///   - targets: ground truth values
///   - beta: threshold after which the loss changes from the squared to the absolute difference
///   - reduction: reduction type
/// - Returns: computed smooth L1 loss
///
/// ### See Also
/// - <doc:losses>
public func smoothL1Loss(
    predictions: MLXArray, targets: MLXArray, beta: Float = 1, reduction: LossReduction = .mean
) -> MLXArray {
    precondition(predictions.shape == targets.shape)

    let diff = abs(predictions - targets)
    let loss = which(diff .< beta, 0.5 * square(diff) / beta, diff - 0.5 * beta)

    return reduction.reduce(loss: loss)
}

/// Computes the triplet loss for a set of anchor, positive, and negative samples. Margin is represented with alpha in the math section.
///
/// - Parameters:
///   - anchors: anchor samples
///   - positives: positive samples
///   - negatives: negative samples
///   - axis: distribution axis
///   - p: norm dree for pairwise distance
///   - margin: margin for the triplet loss
///   - eps: small positive constant to prevent numerical instability
///   - reduction: reduction type
/// - Returns: Computed triplet loss.
///
/// ### See Also
/// - <doc:losses>
public func tripletLoss(
    anchors: MLXArray, positives: MLXArray, negatives: MLXArray, axis: Int = -1, p: Int = 2,
    margin: Float = 1.0, eps: Float = 1e-6, reduction: LossReduction = .none
) -> MLXArray {
    let loss = maximum(
        sqrt(pow(anchors - positives, p).sum(axis: axis) + eps)
            - sqrt(pow(anchors - negatives, p).sum(axis: axis) + eps)
            + margin,
        0
    )

    return reduction.reduce(loss: loss)
}

/// Computes the hinge loss between inputs and targets.
///
/// - Parameters:
///   - inputs: predicted values
///   - targets: target values: -1 or 1
///   - reduction: reduction type
/// - Returns: computed hinge loss
///
/// ### See Also
/// - <doc:losses>
public func hingeLoss(inputs: MLXArray, targets: MLXArray, reduction: LossReduction = .none)
    -> MLXArray
{
    let loss = maximum(1 - inputs * targets, 0)
    return reduction.reduce(loss: loss)
}

/// Computes the Huber loss between inputs and targets.
///
/// - Parameters:
///   - inputs: predicted values
///   - targets: target values
///   - delta: threshold at which to change between L1 and L2 loss
///   - reduction: reduction type
/// - Returns: computed Huber loss
///
/// ### See Also
/// - <doc:losses>
public func huberLoss(
    inputs: MLXArray, targets: MLXArray, delta: Float = 1.0, reduction: LossReduction = .none
) -> MLXArray {
    let errors = inputs - targets
    let absErrors = abs(errors)
    let quadratic = minimum(absErrors, delta)
    let linear = absErrors - quadratic
    let loss = 0.5 * quadratic ** 2 + delta * linear

    return reduction.reduce(loss: loss)
}

/// Computes the log cosh loss between inputs and targets.
///
/// Logcosh acts like L2 loss for small errors, ensuring stable gradients,
/// and like the L1 loss for large errors, reducing sensitivity to outliers. This
/// dual behavior offers a balanced, robust approach for regression tasks.
///
/// - Parameters:
///   - inputs: predicted values
///   - targets: target values
///   - reduction: reduction type
/// - Returns: computed log cosh loss
///
/// ### See Also
/// - <doc:losses>
public func logCoshLoss(inputs: MLXArray, targets: MLXArray, reduction: LossReduction = .none)
    -> MLXArray
{
    let errors = inputs - targets
    let loss = logAddExp(errors, -errors) - log(2.0)

    return reduction.reduce(loss: loss)
}

/// Computes the cosine similarity between the two inputs.
///
/// - Parameters:
///   - x1: first array
///   - x2: second array
///   - axis: embedding axis
///   - eps: minimum value of the denominator used for numerical stability
///   - reduction: reduction type
/// - Returns: computed cosine similarity loss
///
/// ### See Also
/// - <doc:losses>
public func cosineSimilarityLoss(
    x1: MLXArray, x2: MLXArray, axis: Int = 1, eps: Float = 1e-8, reduction: LossReduction = .none
) -> MLXArray {
    func l2Norm(_ a: MLXArray, axis: Int) -> MLXArray {
        if a.dtype.isComplex {
            return sqrt(sum(abs(a) * abs(a), axis: axis))
        } else {
            return sqrt(sum(square(a), axis: axis))
        }
    }

    let x1Norm = l2Norm(x1, axis: axis)
    let x2Norm = l2Norm(x2, axis: axis)

    let loss = sum(x1 * x2, axis: axis) / maximum(x1Norm * x2Norm, eps)

    return reduction.reduce(loss: loss)
}
