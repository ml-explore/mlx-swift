// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// An Elman recurrent layer.
///
/// See ``RNN/init(inputSize:hiddenSize:bias:nonLinearity:)``
open class RNN: Module {

    public let hiddenSize: Int

    public let nonLinearity: (MLXArray, StreamOrDevice) -> MLXArray
    @ParameterInfo(key: "Wxh") public var wxh: MLXArray
    @ParameterInfo(key: "Whh") public var whh: MLXArray
    public let bias: MLXArray?

    /// An Elman recurrent layer.
    ///
    /// The input is a sequence of shape `NLD` or `LD` where:
    ///
    /// * `N` is the optional batch dimension
    /// * `L` is the sequence length
    /// * `D` is the input's feature dimension
    ///
    /// The hidden state `h` has shape `NH` or `H`, depending on
    /// whether the input is batched or not. Returns the hidden state at each
    /// time step, of shape `NLH` or `LH`.
    ///
    /// - Parameters:
    ///   - inputSize: dimension of the input, `D`
    ///   - hiddenSize: dimension of the hidden state, `H`
    ///   - bias: if `true` use a bias
    ///   - nonLinearity: non-linearity to use
    public init(
        inputSize: Int, hiddenSize: Int, bias: Bool = true,
        nonLinearity: @escaping (MLXArray, StreamOrDevice) -> MLXArray = tanh
    ) {
        self.hiddenSize = hiddenSize
        self.nonLinearity = nonLinearity

        let scale = 1 / sqrt(Float(hiddenSize))
        self._wxh.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [hiddenSize, inputSize])
        self._whh.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [hiddenSize, hiddenSize])
        if bias {
            self.bias = MLXRandom.uniform(low: -scale, high: scale, [hiddenSize])
        } else {
            self.bias = nil
        }
    }

    open func callAsFunction(_ x: MLXArray, hidden: MLXArray? = nil) -> MLXArray {
        var x = x

        if let bias {
            x = addMM(bias, x, wxh.T)
        } else {
            x = matmul(x, wxh.T)
        }

        var hidden: MLXArray! = hidden
        var allHidden = [MLXArray]()
        for index in 0 ..< x.dim(-2) {
            if hidden != nil {
                hidden = addMM(x[.ellipsis, index, 0...], hidden, whh.T)
            } else {
                hidden = x[.ellipsis, index, 0...]
            }

            hidden = nonLinearity(hidden, .default)
            allHidden.append(hidden)
        }

        return stacked(allHidden, axis: -2)
    }
}

/// A gated recurrent unit (GRU) RNN layer.
///
/// See ``GRU/init(inputSize:hiddenSize:bias:)``
open class GRU: Module {

    public let hiddenSize: Int

    @ParameterInfo(key: "Wx") public var wx: MLXArray
    @ParameterInfo(key: "Wh") public var wh: MLXArray
    public let b: MLXArray?
    public let bhn: MLXArray?

    /// A gated recurrent unit (GRU) RNN layer.
    ///
    /// The input has shape `NLD` or `LD` where:
    ///
    /// * `N` is the optional batch dimension
    /// * `L` is the sequence length
    /// * `D` is the input's feature dimension
    ///
    /// The hidden state `h` has shape `NH` or `H`, depending on
    /// whether the input is batched or not. Returns the hidden state at each
    /// time step, of shape `NLH` or `LH`.
    ///
    /// - Parameters:
    ///   - inputSize: dimension of the input, `D`
    ///   - hiddenSize: dimension of the hidden state, `H`
    ///   - bias: if `true` use a bias
    public init(inputSize: Int, hiddenSize: Int, bias: Bool = true) {
        self.hiddenSize = hiddenSize
        let scale = 1 / sqrt(Float(hiddenSize))
        self._wx.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [3 * hiddenSize, inputSize])
        self._wh.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [3 * hiddenSize, hiddenSize])
        if bias {
            self.b = MLXRandom.uniform(low: -scale, high: scale, [3 * hiddenSize])
            self.bhn = MLXRandom.uniform(low: -scale, high: scale, [hiddenSize])
        } else {
            self.b = nil
            self.bhn = nil
        }
    }

    open func callAsFunction(_ x: MLXArray, hidden: MLXArray? = nil) -> MLXArray {
        var x = x

        if let b {
            x = addMM(b, x, wx.T)
        } else {
            x = matmul(x, wx.T)
        }

        let x_rz = x[.ellipsis, .stride(to: -hiddenSize)]
        let x_n = x[.ellipsis, .stride(from: -hiddenSize)]

        var hidden: MLXArray! = hidden
        var allHidden = [MLXArray]()

        for index in 0 ..< x.dim(-2) {
            var rz = x_rz[.ellipsis, index, 0...]
            var hProj_n: MLXArray!
            if hidden != nil {
                let hProj = matmul(hidden, wh.T)
                let hProj_rz = hProj[.ellipsis, .stride(to: -hiddenSize)]
                hProj_n = hProj[.ellipsis, .stride(from: -hiddenSize)]

                if let bhn {
                    hProj_n = hProj_n + bhn
                }

                rz = rz + hProj_rz
            }

            rz = sigmoid(rz)

            let parts = split(rz, parts: 2, axis: -1)
            let r = parts[0]
            let z = parts[1]

            var n = x_n[.ellipsis, index, 0...]

            if hidden != nil {
                // Note: xProj_n was computed earlier
                n = n + r * hProj_n
            }
            n = tanh(n)

            if hidden != nil {
                hidden = (1 - z) * n + z * hidden
            } else {
                hidden = (1 - z) * n
            }

            allHidden.append(hidden)
        }

        return stacked(allHidden, axis: -2)
    }
}

/// An LSTM recurrent layer.
///
/// See ``LSTM/init(inputSize:hiddenSize:bias:)``
open class LSTM: Module {

    public let hiddenSize: Int

    @ParameterInfo(key: "Wx") public var wx: MLXArray
    @ParameterInfo(key: "Wh") public var wh: MLXArray
    public let bias: MLXArray?

    /// An LSTM recurrent layer.
    ///
    /// The input has shape `NLD` or `LD` where:
    ///
    /// * `N` is the optional batch dimension
    /// * `L` is the sequence length
    /// * `D` is the input's feature dimension
    ///
    /// The hidden state `h` and cell `c` have shape `NH` or `H`, depending on
    /// whether the input is batched or not. Returns the hidden state and cell state at each
    /// time step, of shape `NLH` or `LH`.
    ///
    /// - Parameters:
    ///   - inputSize: dimension of the input, `D`
    ///   - hiddenSize: dimension of the hidden state, `H`
    ///   - bias: if `true` use a bias
    public init(inputSize: Int, hiddenSize: Int, bias: Bool = true) {
        self.hiddenSize = hiddenSize
        let scale = 1 / sqrt(Float(hiddenSize))
        self._wx.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [4 * hiddenSize, inputSize])
        self._wh.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [4 * hiddenSize, hiddenSize])
        if bias {
            self.bias = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize])
        } else {
            self.bias = nil
        }
    }

    open func callAsFunction(_ x: MLXArray, hidden: MLXArray? = nil, cell: MLXArray? = nil) -> (
        MLXArray, MLXArray
    ) {
        var x = x

        if let bias {
            x = addMM(bias, x, wx.T)
        } else {
            x = matmul(x, wx.T)
        }

        var hidden: MLXArray! = hidden
        var cell: MLXArray! = cell
        var allHidden = [MLXArray]()
        var allCell = [MLXArray]()

        for index in 0 ..< x.dim(-2) {
            var ifgo = x[.ellipsis, index, 0...]
            if hidden != nil {
                ifgo = addMM(ifgo, hidden, wh.T)
            }

            let pieces = split(ifgo, parts: 4, axis: -1)

            let i = sigmoid(pieces[0])
            let f = sigmoid(pieces[1])
            let g = tanh(pieces[2])
            let o = sigmoid(pieces[3])

            if cell != nil {
                cell = f * cell + i * g
            } else {
                cell = i * g
            }
            hidden = o * tanh(cell)

            allCell.append(cell)
            allHidden.append(hidden)
        }

        return (
            stacked(allHidden, axis: -2),
            stacked(allCell, axis: -2)
        )
    }
}
