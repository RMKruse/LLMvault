import eslint from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: [".scratch/", "main.js"] },
  eslint.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ["test/**/*.mjs"],
    languageOptions: { globals: { URL: "readonly" } },
  },
);
