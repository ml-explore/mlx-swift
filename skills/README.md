# MLX Swift Skills

This repo ships two skill definitions:

- **`skills/mlx-swift/`** — Core MLX Swift framework (arrays, ops, NN, optimizers, transforms)
- **`skills/mlx-distributed/`** — MLX Swift Distributed (multi-device communication, tensor parallelism, distributed NN layers)

Each skill has a `SKILL.md` file plus a `references/` folder. The install folder
names match the directory names shown below. If your local copy lives elsewhere,
swap the source paths in the commands.

## Install globally (home directory)

Run these from the repo root, or replace `$(pwd)` with an absolute path.

### Claude Code

```sh
mkdir -p ~/.claude/skills
ln -s "$(pwd)/skills/mlx-swift" ~/.claude/skills/mlx-swift
ln -s "$(pwd)/skills/mlx-distributed" ~/.claude/skills/mlx-distributed
```

### Codex

```sh
mkdir -p ~/.codex/skills
ln -s "$(pwd)/skills/mlx-swift" ~/.codex/skills/mlx-swift
ln -s "$(pwd)/skills/mlx-distributed" ~/.codex/skills/mlx-distributed
```

### Droid

```sh
mkdir -p ~/.agents/skills
ln -s "$(pwd)/skills/mlx-swift" ~/.agents/skills/mlx-swift
ln -s "$(pwd)/skills/mlx-distributed" ~/.agents/skills/mlx-distributed
```

## Install per-project

Create a local skills folder in the project and link the skills there.

### Claude Code

```sh
mkdir -p .claude/skills
ln -s "$(pwd)/skills/mlx-swift" .claude/skills/mlx-swift
ln -s "$(pwd)/skills/mlx-distributed" .claude/skills/mlx-distributed
```

### Codex

```sh
mkdir -p .codex/skills
ln -s "$(pwd)/skills/mlx-swift" .codex/skills/mlx-swift
ln -s "$(pwd)/skills/mlx-distributed" .codex/skills/mlx-distributed
```

### Droid

```sh
mkdir -p .agents/skills
ln -s "$(pwd)/skills/mlx-swift" .agents/skills/mlx-swift
ln -s "$(pwd)/skills/mlx-distributed" .agents/skills/mlx-distributed
```

## Notes

- If your tool caches skills, restart it after installing.
- If you prefer a copy over a symlink, replace `ln -s` with `cp -R`.
