// Copyright © 2024 Apple Inc.

import Foundation

#if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
    enum SwiftPMMetallibResource {
        private static let lock = NSLock()
        #if swift(>=5.10)
            nonisolated(unsafe) private static var didConfigure = false
        #else
            private static var didConfigure = false
        #endif
        private static let bundleName = "mlx-swift_Cmlx.bundle"
        private static let metallibName = "default.metallib"

        static func configureIfNeeded() {
            lock.withLock {
                guard !didConfigure else { return }
                defer { didConfigure = true }

                guard GPU.metallib == nil, let url = findMetallibURL() else { return }
                GPU.metallib = url
            }
        }

        static func findMetallibURL() -> URL? {
            var seen = Set<String>()

            for directory in searchDirectories() {
                var current: URL? = directory
                for _ in 0 ..< 8 {
                    guard let candidateDirectory = current else {
                        break
                    }

                    let key = candidateDirectory.standardizedFileURL.path()
                    if seen.insert(key).inserted,
                        let url = metallibURL(near: candidateDirectory)
                    {
                        return url
                    }

                    let parent = candidateDirectory.deletingLastPathComponent()
                    if parent.path() == candidateDirectory.path() {
                        break
                    }
                    current = parent
                }
            }

            return nil
        }

        static func metallibURL(near directory: URL) -> URL? {
            var bundles = [
                directory.appendingPathComponent(bundleName),
                directory.appendingPathComponent("Resources").appendingPathComponent(bundleName),
                directory.appendingPathComponent("Contents/Resources").appendingPathComponent(
                    bundleName),
            ]
            if directory.lastPathComponent == bundleName {
                bundles.insert(directory, at: 0)
            } else if directory.lastPathComponent == "Resources",
                directory.deletingLastPathComponent().lastPathComponent == "Contents",
                directory.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent
                    == bundleName
            {
                bundles.insert(
                    directory.deletingLastPathComponent().deletingLastPathComponent(), at: 0)
            }
            let candidates = bundles.flatMap { bundle in
                [
                    bundle.appendingPathComponent(metallibName),
                    bundle.appendingPathComponent("Contents/Resources").appendingPathComponent(
                        metallibName),
                ]
            }

            return candidates.first {
                FileManager.default.isReadableFile(atPath: $0.path())
            }
        }

        private static func searchDirectories() -> [URL] {
            var directories = [URL]()
            var seen = Set<String>()

            func add(_ url: URL?) {
                guard let url else {
                    return
                }
                let standardized = url.standardizedFileURL
                if seen.insert(standardized.path()).inserted {
                    directories.append(standardized)
                }
            }

            add(Bundle.main.resourceURL)
            add(Bundle.main.bundleURL)
            add(Bundle.main.executableURL?.deletingLastPathComponent())

            for bundle in Bundle.allBundles {
                add(bundle.resourceURL)
                add(bundle.bundleURL)
                add(bundle.executableURL?.deletingLastPathComponent())
            }

            for framework in Bundle.allFrameworks {
                add(framework.resourceURL)
                add(framework.bundleURL)
                add(framework.executableURL?.deletingLastPathComponent())
            }

            if let executablePath = CommandLine.arguments.first, !executablePath.isEmpty {
                add(URL(fileURLWithPath: executablePath).deletingLastPathComponent())
            }

            return directories
        }
    }
#endif
