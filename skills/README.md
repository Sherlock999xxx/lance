# Skills

This directory contains Codex-compatible skills for the Lance project.

Each skill is a folder that contains a required `SKILL.md` (with YAML frontmatter) and optional `scripts/`, `references/`, and `assets/`.

## Install (npx skills)

If you use `skills.sh`, install from GitHub:

```bash
npx skills add lance-format/lance --skill lance-user-guide
```

Install globally (user-level):

```bash
npx skills add lance-format/lance --skill lance-user-guide -g
```

List available skills in this repository:

```bash
npx skills add lance-format/lance --list
```

## Install (manual copy)

Codex typically loads skills from:

- Project: `.codex/skills/<skill-name>/`
- Global: `~/.codex/skills/<skill-name>/`

Install into the current repository:

```bash
mkdir -p .codex/skills
cp -R skills/lance-user-guide .codex/skills/
```

Install globally:

```bash
mkdir -p ~/.codex/skills
cp -R skills/lance-user-guide ~/.codex/skills/
```

Restart Codex after installing or updating skills.
