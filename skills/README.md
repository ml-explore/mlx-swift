# MLX Swift skill

This repo ships an MLX Swift skill definition under `skills/mlx-swift/` (the `skill.md`
file plus `references/`). The install folder name can be `mlx-swift`, as shown below.
If your local copy lives at `skills/mlx-swift`, just swap the source path in the
commands.

## Install globally (home directory)

Run these from the repo root, or replace `$(pwd)` with an absolute path.

### Claude Code

```sh
mkdir -p ~/.claude/skills
ln -s "$(pwd)/skills/mlx-swift" ~/.claude/skills/mlx-swift
```

### Codex

```sh
mkdir -p ~/.codex/skills
ln -s "$(pwd)/skills/mlx-swift" ~/.codex/skills/mlx-swift
```

### Droid

```sh
mkdir -p ~/.agents/skills
ln -s "$(pwd)/skills/mlx-swift" ~/.agents/skills/mlx-swift
```

## Install per-project

Create a local skills folder in the project and link the skill there.

### Claude Code

```sh
mkdir -p .claude/skills
ln -s "$(pwd)/skills/mlx-swift" .claude/skills/mlx-swift
```

### Codex

```sh
mkdir -p .codex/skills
ln -s "$(pwd)/skills/mlx-swift" .codex/skills/mlx-swift
```

### Droid

```sh
mkdir -p .agents/skills
ln -s "$(pwd)/skills/mlx-swift" .agents/skills/mlx-swift
```

## Notes

- If your tool caches skills, restart it after installing.
- If you prefer a copy over a symlink, replace `ln -s` with `cp -R`.
