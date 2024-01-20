import Foundation

// Module
// Linear
// RoPE
// RMSNorm
// treemap -- used to apply a dtype to parameters

open class Module : CustomStringConvertible {
    
    var training = false
    var noGrad = Set<String>()
    
    public private(set) var children = [(String, Module)]()
    
    public init() {
        buildChildren()
    }
    
    public func buildChildren() {
        let m = Mirror(reflecting: self)
        for c in m.children {
            if let value = c.value as? Module, let label = c.label {
                children.append((label, value))
            }
        }
    }
    
    public func describeParameters(_ indent: Int) -> String {
        var result = [String]()
        
        let m = Mirror(reflecting: self)
        for c in m.children {
            if c.value is Module { continue }
            
            if let label = c.label {
                let value = String(describing: c.value)
                result.append("\(label)=\(value)")
            }
        }
        
        if result.isEmpty {
            return ""
        } else {
            return "(\(result.joined(separator: ", ")))"
        }
    }
    
    public func description(_ indent: Int) -> String {
        var result = ""
        
        result += "\(String(describing: type(of: self)))\(describeParameters(indent))"
        
        if !children.isEmpty {
            let indentString = String(repeating: " ", count: indent)
            
            result += " {\n"
            for (k, v) in children {
                result += "\(indentString)  \(k): \(v.description(indent + 2)),\n"
            }
            result += "\(indentString)}"
        }
        
        return result
    }
    
    public var description: String {
        description(0)
    }
}
