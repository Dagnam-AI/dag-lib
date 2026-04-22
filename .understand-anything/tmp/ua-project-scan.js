#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");

const projectRoot = process.argv[2];
const outputPath = process.argv[3];

if (!projectRoot || !outputPath) {
  process.stderr.write("Usage: node ua-project-scan.js <project-root> <output-path>\n");
  process.exit(1);
}
if (!fs.existsSync(projectRoot)) {
  process.stderr.write("Cannot access directory: " + projectRoot + "\n");
  process.exit(1);
}

// Step 1: File Discovery
let allFiles = [];
try {
  const result = spawnSync("git", ["ls-files"], { cwd: projectRoot, encoding: "utf8" });
  if (result.status === 0 && result.stdout.trim()) {
    allFiles = result.stdout.trim().split("\n").filter(Boolean);
  } else throw new Error("git ls-files failed");
} catch (e) {
  function walkDir(dir, base) {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      const rel = path.relative(base, full).replace(/\\/g, "/");
      if (entry.isDirectory()) walkDir(full, base);
      else allFiles.push(rel);
    }
  }
  walkDir(projectRoot, projectRoot);
}
allFiles = allFiles.map(f => f.replace(/\\/g, "/"));

// Step 2: Exclusion Filtering
const EXCL_DIRS = new Set(["node_modules",".git","vendor","venv",".venv","__pycache__","dist","build","out","coverage",".next",".cache",".turbo","target","obj",".idea",".vscode",".hypothesis",".pytest_cache",".ruff_cache",".mypy_cache"]);
const EXCL_FILES = new Set(["LICENSE",".gitignore",".editorconfig",".prettierrc","package-lock.json","yarn.lock","pnpm-lock.yaml","uv.lock"]);
const EXCL_EXT = new Set([".png",".jpg",".jpeg",".gif",".svg",".ico",".woff",".woff2",".ttf",".eot",".mp3",".mp4",".pdf",".zip",".tar",".gz",".log"]);

function isExcluded(filePath) {
  const parts = filePath.split("/");
  const filename = parts[parts.length - 1];
  for (let i = 0; i < parts.length - 1; i++) {
    if (EXCL_DIRS.has(parts[i])) return true;
  }
  if (EXCL_FILES.has(filename)) return true;
  const ext = path.extname(filename).toLowerCase();
  if (EXCL_EXT.has(ext)) return true;
  if (filename.endsWith(".lock")) return true;
  if (filename.endsWith(".min.js") || filename.endsWith(".min.css") || filename.endsWith(".map")) return true;
  if (filename.includes(".generated.")) return true;
  if (filename.startsWith(".eslintrc")) return true;
  return false;
}

const filteredFiles = allFiles.filter(f => !isExcluded(f));

// Step 2.5: .understandignore
let filteredByIgnore = 0;
const understandIgnorePaths = [
  path.join(projectRoot, ".understand-anything", ".understandignore"),
  path.join(projectRoot, ".understandignore")
];
let ignorePatterns = [];
for (const igPath of understandIgnorePaths) {
  if (fs.existsSync(igPath)) {
    const lines = fs.readFileSync(igPath, "utf8").split("\n").map(l => l.trim()).filter(l => l && !l.startsWith("#"));
    ignorePatterns.push(...lines);
  }
}

let finalFiles = filteredFiles;
if (ignorePatterns.length > 0) {
  function matchesPattern(filePath, pattern) {
    if (pattern.startsWith("!")) return false;
    if (pattern.endsWith("/")) {
      const dir = pattern.slice(0, -1);
      return filePath.split("/").some(seg => seg === dir) || filePath.startsWith(dir + "/");
    }
    if (pattern.startsWith("*.")) {
      return filePath.endsWith(pattern.slice(1));
    }
    return filePath === pattern || filePath.startsWith(pattern + "/") || filePath.split("/").some(seg => seg === pattern);
  }
  const negations = ignorePatterns.filter(p => p.startsWith("!")).map(p => p.slice(1));
  const positives = ignorePatterns.filter(p => !p.startsWith("!"));
  const beforeIgnore = filteredFiles.length;
  finalFiles = filteredFiles.filter(f => {
    const excluded = positives.some(p => matchesPattern(f, p));
    if (!excluded) return true;
    return negations.some(p => matchesPattern(f, p));
  });
  filteredByIgnore = beforeIgnore - finalFiles.length;
}

// Step 3: Language Detection
const EXT_TO_LANG = {".ts":"typescript",".tsx":"typescript",".js":"javascript",".jsx":"javascript",".py":"python",".go":"go",".rs":"rust",".java":"java",".rb":"ruby",".cpp":"cpp",".cc":"cpp",".cxx":"cpp",".h":"cpp",".hpp":"cpp",".c":"c",".cs":"csharp",".swift":"swift",".kt":"kotlin",".php":"php",".vue":"vue",".svelte":"svelte",".sh":"shell",".bash":"shell",".ps1":"shell",".bat":"shell",".md":"markdown",".rst":"markdown",".yaml":"yaml",".yml":"yaml",".json":"json",".toml":"toml",".sql":"sql",".graphql":"graphql",".gql":"graphql",".proto":"protobuf",".tf":"terraform",".tfvars":"terraform",".html":"html",".htm":"html",".css":"css",".scss":"css",".sass":"css",".less":"css",".xml":"xml",".cfg":"config",".ini":"config",".env":"config"};
const BASENAME_TO_LANG = {"Dockerfile":"dockerfile","Makefile":"makefile","Jenkinsfile":"jenkinsfile"};

function detectLanguage(filePath) {
  const basename = path.basename(filePath);
  if (BASENAME_TO_LANG[basename]) return BASENAME_TO_LANG[basename];
  const ext = path.extname(basename).toLowerCase();
  return EXT_TO_LANG[ext] || "unknown";
}

// Step 4: File Category
function detectCategory(filePath) {
  const basename = path.basename(filePath);
  const ext = path.extname(basename).toLowerCase();
  if (basename === "Dockerfile" || basename.startsWith("docker-compose")) return "infra";
  if ([".tf",".tfvars"].includes(ext)) return "infra";
  if (["Makefile","Jenkinsfile","Procfile","Vagrantfile"].includes(basename)) return "infra";
  if (filePath.includes(".github/workflows/")) return "infra";
  if (basename === ".gitlab-ci.yml") return "infra";
  if (filePath.includes("k8s/") || filePath.includes("kubernetes/")) return "infra";
  if (filePath.endsWith(".k8s.yaml") || filePath.endsWith(".k8s.yml")) return "infra";
  if ([".md",".rst",".txt"].includes(ext)) return "docs";
  if ([".yaml",".yml",".json",".toml",".xml",".cfg",".ini",".env"].includes(ext)) return "config";
  if (["tsconfig.json","package.json","pyproject.toml","Cargo.toml","go.mod"].includes(basename)) return "config";
  if ([".sql",".graphql",".gql",".proto",".prisma",".csv"].includes(ext)) return "data";
  if (basename.endsWith(".schema.json")) return "data";
  if ([".sh",".bash",".ps1",".bat"].includes(ext)) return "script";
  if ([".html",".htm",".css",".scss",".sass",".less"].includes(ext)) return "markup";
  return "code";
}

// Step 5: Line Counting
function countLines(files) {
  const counts = {};
  for (const f of files) {
    try {
      const content = fs.readFileSync(path.join(projectRoot, f), "utf8");
      counts[f] = content.split("\n").length;
    } catch (err) { counts[f] = 0; }
  }
  return counts;
}

// Step 6: Framework Detection
let frameworks = [];
let projectName = path.basename(projectRoot);
let rawDescription = "";
let readmeHead = "";

const readmePath = path.join(projectRoot, "README.md");
if (fs.existsSync(readmePath)) {
  const rc = fs.readFileSync(readmePath, "utf8");
  readmeHead = rc.split("\n").slice(0, 10).join("\n");
}

const pyprojectPath = path.join(projectRoot, "pyproject.toml");
if (fs.existsSync(pyprojectPath)) {
  const content = fs.readFileSync(pyprojectPath, "utf8");
  const nameMatch = content.match(/^\s*name\s*=\s*"([^"]+)"/m);
  if (nameMatch) projectName = nameMatch[1];
  const descMatch = content.match(/^\s*description\s*=\s*"([^"]+)"/m);
  if (descMatch) rawDescription = descMatch[1];
  const pyFws = ["django","djangorestframework","fastapi","flask","sqlalchemy","alembic","celery","pydantic","uvicorn","gunicorn","aiohttp","tornado","starlette","pytest","hypothesis","channels"];
  for (const fw of pyFws) {
    if (content.toLowerCase().includes(fw)) {
      const fwName = fw.charAt(0).toUpperCase() + fw.slice(1);
      if (!frameworks.includes(fwName)) frameworks.push(fwName);
    }
  }
}

if (finalFiles.some(f => path.basename(f) === "Dockerfile")) frameworks.push("Docker");
if (finalFiles.some(f => f.includes("docker-compose"))) frameworks.push("Docker Compose");
if (finalFiles.some(f => f.endsWith(".tf"))) frameworks.push("Terraform");
if (finalFiles.some(f => f.includes(".github/workflows/"))) frameworks.push("GitHub Actions");
if (finalFiles.some(f => path.basename(f) === ".gitlab-ci.yml")) frameworks.push("GitLab CI");
if (finalFiles.some(f => path.basename(f) === "Jenkinsfile")) frameworks.push("Jenkins");
frameworks = [...new Set(frameworks)];

// Step 7: Complexity
function estimateComplexity(count) {
  if (count <= 30) return "small";
  if (count <= 150) return "moderate";
  if (count <= 500) return "large";
  return "very-large";
}

const lineCounts = countLines(finalFiles);
const fileSet = new Set(finalFiles);

const files = finalFiles.map(f => ({
  path: f,
  language: detectLanguage(f),
  sizeLines: lineCounts[f] || 0,
  fileCategory: detectCategory(f),
})).sort((a, b) => a.path.localeCompare(b.path));

const languages = [...new Set(files.map(f => f.language).filter(l => l !== "unknown"))].sort();

// Step 9: Import Resolution
function resolveRelativeImport(importPath, importingFile, fileSet) {
  const importingDir = path.dirname(importingFile).replace(/\\/g, "/");
  let resolved = path.posix.normalize(importingDir + "/" + importPath);
  if (fileSet.has(resolved)) return resolved;
  const extensions = [".ts",".tsx",".js",".jsx","/index.ts","/index.js","/index.tsx","/index.jsx",".py",".go",".rs",".rb"];
  for (const ext of extensions) {
    const candidate = resolved + ext;
    if (fileSet.has(candidate)) return candidate;
  }
  return null;
}

function extractImports(filePath, content, language) {
  const resolved = [];
  if (language === "typescript" || language === "javascript") {
    const patterns = [
      /import\s+(?:.*?\s+from\s+)?["']([.][^"']+)["']/g,
      /require\s*\(\s*["']([.][^"']+)["']\s*\)/g,
    ];
    for (const pattern of patterns) {
      let match;
      while ((match = pattern.exec(content)) !== null) {
        const r = resolveRelativeImport(match[1], filePath, fileSet);
        if (r) resolved.push(r);
      }
    }
  } else if (language === "python") {
    const relPattern = /^from\s+(\.+)([\w.]+)?\s+import/gm;
    let match;
    while ((match = relPattern.exec(content)) !== null) {
      const dots = match[1].length;
      const rest = (match[2] || "").replace(/\./g, "/");
      let base = path.dirname(filePath).replace(/\\/g, "/");
      for (let i = 1; i < dots; i++) base = path.posix.dirname(base);
      const candidate = rest ? base + "/" + rest : base;
      const norm1 = (candidate + ".py").replace(/\/\/+/g, "/");
      const norm2 = (candidate + "/__init__.py").replace(/\/\/+/g, "/");
      if (fileSet.has(norm1)) resolved.push(norm1);
      else if (fileSet.has(norm2)) resolved.push(norm2);
    }
  } else if (language === "ruby") {
    const pattern = /require_relative\s+["']([^"']+)["']/g;
    let match;
    while ((match = pattern.exec(content)) !== null) {
      const r = resolveRelativeImport(match[1], filePath, fileSet);
      if (r) resolved.push(r);
    }
  }
  return [...new Set(resolved)];
}

const importMap = {};
for (const file of files) {
  if (file.fileCategory !== "code") { importMap[file.path] = []; continue; }
  try {
    const content = fs.readFileSync(path.join(projectRoot, file.path), "utf8");
    importMap[file.path] = extractImports(file.path, content, file.language);
  } catch (e) { importMap[file.path] = []; }
}

const output = {
  scriptCompleted: true,
  name: projectName,
  rawDescription,
  readmeHead,
  languages,
  frameworks,
  files,
  totalFiles: files.length,
  filteredByIgnore,
  estimatedComplexity: estimateComplexity(files.length),
  importMap,
};

fs.writeFileSync(outputPath, JSON.stringify(output, null, 2), "utf8");
process.exit(0);
