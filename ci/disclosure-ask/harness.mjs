// Offline harness for .github/workflows/disclosure-ask.yml.
//
// Pulls the inline github-script out of the workflow file, so there is no
// second copy of the logic to drift, and runs it against a fake GitHub client
// for each case the design has to get right. Posting nothing is as much a case
// as posting once. Run from the repository root:
//
//     node ci/disclosure-ask/harness.mjs
//
// A path to another copy of the workflow may be given as the only argument.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const here = dirname(fileURLToPath(import.meta.url));
const yamlPath = process.argv[2] || resolve(here, '../../.github/workflows/disclosure-ask.yml');
const text = readFileSync(yamlPath, 'utf8');
// Pull the `script: |` block: every line after it indented deeper than `script:`.
const lines = text.split('\n');
const start = lines.findIndex((l) => /^\s*script: \|\s*$/.test(l));
if (start < 0) throw new Error('script block not found');
const indent = lines[start].match(/^\s*/)[0].length + 2;
const body = [];
for (const l of lines.slice(start + 1)) {
  if (l.trim() === '' ) { body.push(''); continue; }
  if (l.match(/^\s*/)[0].length < indent) break;
  body.push(l.slice(indent));
}
const script = body.join('\n');

function fake({ pr, commits = [], comments = [], permission = null, permissionStatus = 200 }) {
  const log = [];
  const posted = [];
  const github = {
    rest: {
      pulls: {
        get: async () => ({ data: pr }),
        listCommits: 'listCommits',
      },
      issues: {
        listComments: 'listComments',
        createComment: async ({ body }) => { posted.push(body); return {}; },
      },
      repos: {
        getCollaboratorPermissionLevel: async () => {
          if (permissionStatus !== 200) { const e = new Error('nope'); e.status = permissionStatus; throw e; }
          return { data: permission };
        },
      },
    },
    paginate: async (which) => (which === 'listCommits' ? commits : comments),
  };
  const core = {
    info: (m) => log.push(m),
    setFailed: (m) => log.push('FAILED: ' + m),
  };
  return { github, core, log, posted };
}

const base = {
  number: 42,
  user: { login: 'newcomer', type: 'User' },
  labels: [],
  author_association: 'FIRST_TIME_CONTRIBUTOR',
  body: 'Fixes #1.\n\nSome description.',
};
const commit = (message, sha = 'abcdef1234567') => ({ sha, commit: { message } });
const external = { permission: { permission: 'read', role_name: 'read' } };

const cases = [
  ['external, no line: asks', { pr: base, ...external }, { asks: true }],
  ['repeat contributor: asks without welcome', { pr: { ...base, author_association: 'CONTRIBUTOR' }, ...external }, { asks: true, noWelcome: true }],
  ['bot by type', { pr: { ...base, user: { login: 'dependabot[bot]', type: 'Bot' } } }, { asks: false, why: 'bot' }],
  ['automated label', { pr: { ...base, labels: [{ name: 'automated' }] } }, { asks: false, why: 'labelled' }],
  ['member by association', { pr: { ...base, author_association: 'MEMBER' } }, { asks: false, why: 'association' }],
  ['private member with write (association NONE)', { pr: { ...base, author_association: 'NONE' }, permission: { permission: 'write', role_name: 'write' } }, { asks: false, why: 'holds write' }],
  ['maintain role', { pr: { ...base, author_association: 'NONE' }, permission: { permission: 'write', role_name: 'maintain' } }, { asks: false, why: 'holds maintain' }],
  ['permission lookup 404 treated as external', { pr: base, permissionStatus: 404 }, { asks: true }],
  ['permission lookup 500 propagates', { pr: base, permissionStatus: 500 }, { throws: true }],
  ['line in description', { pr: { ...base, body: 'Text\n\nAssisted-by: Claude Code (Claude Fable 5.1)\n' }, ...external }, { asks: false, why: 'description carries' }],
  ['Assisted-by: none counts', { pr: { ...base, body: 'Assisted-by: none' }, ...external }, { asks: false, why: 'description carries' }],
  ['template left empty: asks', { pr: { ...base, body: '<!-- Assisted-by: <tool> (<model>) or Assisted-by: none -->\nAssisted-by:\n\nMy change.' }, ...external }, { asks: true }],
  ['key only inside HTML comment: asks', { pr: { ...base, body: '<!-- Assisted-by: none -->\nMy change.' }, ...external }, { asks: true }],
  ['key mid-line in prose: asks', { pr: { ...base, body: 'Please add an Assisted-by: line later.' }, ...external }, { asks: true }],
  ['key in inline code: asks', { pr: { ...base, body: 'See `Assisted-by: none` in the guide.' }, ...external }, { asks: true }],
  ['line in a commit message', { pr: base, commits: [commit('fix thing'), commit('more\n\nAssisted-by: none')], ...external }, { asks: false, why: 'commit' }],
  ['Claude Code co-author trailer', { pr: base, commits: [commit('fix\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>')], ...external }, { asks: false, why: 'tool co-author' }],
  ['Copilot co-author trailer', { pr: base, commits: [commit('fix\n\nCo-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>')], ...external }, { asks: false, why: 'tool co-author' }],
  ['human co-author trailer: asks', { pr: base, commits: [commit('fix\n\nCo-authored-by: Jane Doe <jane@example.org>')], ...external }, { asks: true }],
  ['Claude Code footer only: asks', { pr: { ...base, body: 'Change.\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)' }, ...external }, { asks: true }],
  ['already asked (marker on page 2)', { pr: base, comments: [...Array(31).fill({ body: 'chatter' }), { body: '<!-- qep5-disclosure-ask v1 -->\nold ask' }], ...external }, { asks: false, why: 'already asked' }],
  ['dispatch with number', { pr: null, dispatch: '42', fetched: base, ...external }, { asks: true }],
  ['dispatch with bad number', { pr: null, dispatch: 'x' }, { asks: false, why: 'FAILED' }],
  ['dispatch dry run: logs, does not post', { pr: null, dispatch: '42', dryRun: 'true', fetched: base, ...external }, { asks: false, why: 'would ask' }],
];

let failures = 0;
for (const [name, setup, expect] of cases) {
  const f = fake({ ...setup, pr: setup.pr || setup.fetched });
  const context = {
    repo: { owner: 'QuantEcon', repo: 'QuantEcon.py' },
    payload: setup.pr ? { pull_request: setup.pr } : { inputs: { pull_request: setup.dispatch, dry_run: setup.dryRun } },
  };
  let threw = null;
  try {
    await new AsyncFunction('github', 'context', 'core', script)(f.github, context, f.core);
  } catch (e) { threw = e; }
  const asked = f.posted.length === 1;
  const problems = [];
  if (expect.throws) { if (!threw) problems.push('expected a throw'); }
  else if (threw) problems.push('threw: ' + threw.message);
  else {
    if (asked !== !!expect.asks) problems.push(`asks=${asked}, expected ${!!expect.asks}`);
    if (expect.why && !f.log.some((m) => m.includes(expect.why))) problems.push(`log lacks "${expect.why}": ${f.log.join(' | ')}`);
    if (asked && expect.noWelcome && f.posted[0].includes('and welcome to')) problems.push('unexpected welcome');
    if (asked && !expect.noWelcome && !f.posted[0].includes('and welcome to')) problems.push('missing welcome');
    if (asked && !f.posted[0].startsWith('<!-- qep5-disclosure-ask v1 -->')) problems.push('marker not first');
  }
  console.log((problems.length ? 'FAIL ' : 'ok   ') + name + (problems.length ? '\n     ' + problems.join('\n     ') : ''));
  failures += problems.length ? 1 : 0;
}
console.log(`\n${cases.length - failures}/${cases.length} passed`);
if (failures) process.exit(1);
// Print the comment once so the wording can be read.
const f = fake({ pr: base, ...external });
await new AsyncFunction('github', 'context', 'core', script)(f.github, { repo: { owner: 'QuantEcon', repo: 'QuantEcon.py' }, payload: { pull_request: base } }, f.core);
console.log('\n--- comment as posted ---\n' + f.posted[0]);
