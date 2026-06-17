import ArgumentParser

@main
struct Encuda: ParsableCommand {
    static let configuration = CommandConfiguration(
        subcommands: [Compile.self, Link.self]
    )
}
