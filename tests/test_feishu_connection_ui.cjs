const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../frontend/app.js'), 'utf8');
const start = source.indexOf('async function startFeishuEventConnection()');
const end = source.indexOf('function feishuActionLabel(', start);
assert.ok(start >= 0 && end > start);

for (const [name, response, expected] of [
    ['connected', {success: true, message: '已建立'}, '成功：已建立'],
    ['still connecting', {success: false, status: {running: true, state: 'connecting'}, message: '正在建立'}, '连接中：正在建立'],
    ['closed event loop', {success: false, status: {running: false, state: 'error', last_error: 'Event loop is closed'}, message: '正在建立'}, '连接失败：Event loop is closed'],
    ['failed live worker', {success: false, status: {running: true, state: 'error'}, message: '连接异常'}, '连接失败：连接异常'],
    ['disconnected', {success: false, status: {running: false, state: 'disconnected'}, message: '已断开'}, '连接失败：已断开'],
]) {
    test(name, async () => {
        let refreshed = false;
        const context = vm.createContext({
            API_BASE: '/api', feishuMsg: {value: ''}, feishuWorkspaceLoading: {value: false},
            apiFetch: async () => ({ok: true, json: async () => response}),
            loadFeishuWorkspace: async () => {refreshed = true;},
        });
        vm.runInContext(source.slice(start, end), context);
        await context.startFeishuEventConnection();
        assert.equal(context.feishuMsg.value, expected);
        assert.equal(context.feishuWorkspaceLoading.value, false);
        assert.equal(refreshed, true);
    });
}
