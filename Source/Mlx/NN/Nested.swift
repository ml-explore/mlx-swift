import Foundation

public indirect enum Nested<T> {
    case value(T)
    case values(T)
    case nested(Nested<T>)
}
