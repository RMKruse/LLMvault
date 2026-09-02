import { defineConfig, globalIgnores } from "eslint/config";
import obsidianmd from "eslint-plugin-obsidianmd";

export default defineConfig(
  globalIgnores([
    ".scratch/",
    "esbuild.config.mjs",
    "evaluation/",
    "main.js",
    "package-lock.json",
    "package.json",
    "release-candidate/**",
    "scripts/",
    "test/",
    "tsconfig.json",
    "versions.json",
  ]),
  ...obsidianmd.configs.recommended,
  {
    languageOptions: {
      parserOptions: {
        projectService: { allowDefaultProject: ["eslint.config.mjs"] },
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      "obsidianmd/ui/sentence-case": ["warn", {
        acronyms: ["OS", "SSD"],
        brands: [
          "Grounded Answer",
          "LLMvault",
          "Local Model",
          "Local Models",
          "Local Processing",
          "Ollama",
          "Vault Chat",
          "Vault Content",
        ],
        enforceCamelCaseLower: true,
      }],
    },
  },
);
