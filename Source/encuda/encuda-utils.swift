import Foundation

func searchForCommand(_ name: String) -> URL? {
    let path = ProcessInfo.processInfo.environment["PATH"] ?? ""
    for folder in path.split(separator: ":") {
        let url = URL(fileURLWithPath: String(folder)).appendingPathComponent(name)
        if FileManager.default.isExecutableFile(atPath: url.path) {
            return url
        }
    }
    return nil
}

extension Process {
    /**
     * When running on aarch64 Ubuntu 24 docker on Mac, it happens that
     * the `waitUntilExit()` method never returns.
     * We adopted this workaround.
     */
    func waitUntilExitWorkaround() {
        while isRunning {
            Thread.sleep(forTimeInterval: 0.05)
        }
    }
}
