#!/usr/bin/env node
/**
 * Windows-compatible wrapper that replicates extract-structure.mjs logic,
 * but uses pathToFileURL() for dynamic imports on Windows.
 */

import { createRequire } from 'node:module';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { readFileSync, writeFileSync } from 'node:fs';

// Hardcoded plugin root for this environment (passed via env or known).
const pluginRoot = 'C:/Users/angad/.claude/plugins/cache/understand-anything/understand-anything/2.3.1';
const require = createRequire(resolve(pluginRoot, 'package.json'));

let core;
try {
  const resolved = require.resolve('@understand-anything/core');
  core = await import(pathToFileURL(resolved).href);
} catch (e1) {
  try {
    const fallback = resolve(pluginRoot, 'packages/core/dist/index.js');
    core = await import(pathToFileURL(fallback).href);
  } catch (e2) {
    process.stderr.write(`Failed to load core: ${e1.message} / ${e2.message}\n`);
    process.exit(1);
  }
}

const { TreeSitterPlugin, PluginRegistry, builtinLanguageConfigs, registerAllParsers } = core;

const [,, inputPath, outputPath] = process.argv;
if (!inputPath || !outputPath) {
  process.stderr.write('Usage: node extract-structure-win.mjs <input.json> <output.json>\n');
  process.exit(1);
}

async function main() {
  const inputRaw = readFileSync(inputPath, 'utf-8');
  const input = JSON.parse(inputRaw);
  const { projectRoot, batchFiles, batchImportData } = input;

  if (!projectRoot || !Array.isArray(batchFiles)) {
    throw new Error('Invalid input: must contain projectRoot and batchFiles array');
  }

  const tsConfigs = builtinLanguageConfigs.filter(c => c.treeSitter);
  const tsPlugin = new TreeSitterPlugin(tsConfigs);
  await tsPlugin.init();

  const registry = new PluginRegistry();
  registry.register(tsPlugin);
  registerAllParsers(registry);

  const results = [];
  const filesSkipped = [];

  for (const file of batchFiles) {
    const absolutePath = join(projectRoot, file.path);

    let content;
    try {
      content = readFileSync(absolutePath, 'utf-8');
    } catch (err) {
      filesSkipped.push(file.path);
      continue;
    }

    const lines = content.split('\n');
    const totalLines = lines.length;
    const nonEmptyLines = lines.filter(l => l.trim().length > 0).length;

    let analysis = null;
    try {
      analysis = registry.analyzeFile(file.path, content);
    } catch {}

    let callGraph = null;
    if (file.fileCategory === 'code' || file.fileCategory === 'script') {
      try {
        const cg = registry.extractCallGraph(file.path, content);
        if (cg && cg.length > 0) {
          callGraph = cg.map(entry => ({
            caller: entry.caller,
            callee: entry.callee,
            lineNumber: entry.lineNumber,
          }));
        }
      } catch {}
    }

    const result = buildResult(file, totalLines, nonEmptyLines, analysis, callGraph, batchImportData);
    results.push(result);
  }

  const output = {
    scriptCompleted: true,
    filesAnalyzed: results.length,
    filesSkipped,
    results,
  };

  writeFileSync(outputPath, JSON.stringify(output, null, 2), 'utf-8');
}

function buildResult(file, totalLines, nonEmptyLines, analysis, callGraph, batchImportData) {
  const base = {
    path: file.path,
    language: file.language,
    fileCategory: file.fileCategory,
    totalLines,
    nonEmptyLines,
  };

  if (!analysis) {
    base.metrics = {};
    return base;
  }

  if (analysis.functions && analysis.functions.length > 0) {
    base.functions = analysis.functions.map(fn => ({
      name: fn.name,
      startLine: fn.lineRange[0],
      endLine: fn.lineRange[1],
      params: fn.params || [],
    }));
  }

  if (analysis.classes && analysis.classes.length > 0) {
    base.classes = analysis.classes.map(cls => ({
      name: cls.name,
      startLine: cls.lineRange[0],
      endLine: cls.lineRange[1],
      methods: cls.methods || [],
      properties: cls.properties || [],
    }));
  }

  if (analysis.exports && analysis.exports.length > 0) {
    base.exports = analysis.exports.map(exp => ({
      name: exp.name,
      line: exp.lineNumber,
      isDefault: false,
    }));
  }

  if (analysis.sections && analysis.sections.length > 0) {
    base.sections = analysis.sections.map(s => ({
      heading: s.name,
      level: s.level,
      line: s.lineRange[0],
    }));
  }

  if (analysis.definitions && analysis.definitions.length > 0) {
    base.definitions = analysis.definitions.map(d => ({
      name: d.name,
      kind: d.kind,
      fields: d.fields || [],
      startLine: d.lineRange[0],
      endLine: d.lineRange[1],
    }));
  }

  if (analysis.services && analysis.services.length > 0) {
    base.services = analysis.services.map(s => ({
      name: s.name,
      image: s.image,
      ports: s.ports || [],
      ...(s.lineRange ? { startLine: s.lineRange[0], endLine: s.lineRange[1] } : {}),
    }));
  }

  if (analysis.endpoints && analysis.endpoints.length > 0) {
    base.endpoints = analysis.endpoints.map(e => ({
      method: e.method,
      path: e.path,
      startLine: e.lineRange[0],
      endLine: e.lineRange[1],
    }));
  }

  if (analysis.steps && analysis.steps.length > 0) {
    base.steps = analysis.steps.map(s => ({
      name: s.name,
      startLine: s.lineRange[0],
      endLine: s.lineRange[1],
    }));
  }

  if (analysis.resources && analysis.resources.length > 0) {
    base.resources = analysis.resources.map(r => ({
      name: r.name,
      kind: r.kind,
      startLine: r.lineRange[0],
      endLine: r.lineRange[1],
    }));
  }

  if (callGraph && callGraph.length > 0) {
    base.callGraph = callGraph;
  }

  const metrics = {};
  const importPaths = batchImportData?.[file.path];
  if (importPaths) {
    metrics.importCount = importPaths.length;
  } else if (analysis.imports) {
    metrics.importCount = analysis.imports.length;
  }
  if (analysis.exports) metrics.exportCount = analysis.exports.length;
  if (analysis.functions) metrics.functionCount = analysis.functions.length;
  if (analysis.classes) metrics.classCount = analysis.classes.length;
  if (analysis.sections) metrics.sectionCount = analysis.sections.length;
  if (analysis.definitions) metrics.definitionCount = analysis.definitions.length;
  if (analysis.services) metrics.serviceCount = analysis.services.length;
  if (analysis.endpoints) metrics.endpointCount = analysis.endpoints.length;
  if (analysis.steps) metrics.stepCount = analysis.steps.length;
  if (analysis.resources) metrics.resourceCount = analysis.resources.length;

  base.metrics = metrics;

  return base;
}

try {
  await main();
} catch (err) {
  process.stderr.write(`extract-structure-win.mjs failed: ${err.message}\n${err.stack}\n`);
  process.exit(1);
}
