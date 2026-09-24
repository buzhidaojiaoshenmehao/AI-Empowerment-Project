/**
 * AI 赋能项目知识沉淀系统 - 前端逻辑
 * 支持文档管理、RAG对话、知识图谱、飞书集成、离职交接
 */
const APP_ORIGIN = window.location.protocol === 'file:' ? 'http://127.0.0.1:8000' : window.location.origin;
const API_BASE = `${APP_ORIGIN}/api`;
const GRAPH_PALETTE = ['#3f63d8', '#138a78', '#d97706', '#c2415b', '#7c5ac9', '#1676b9', '#a95325'];
const nativeFetch = window.fetch.bind(window);

function readCookie(name) {
    const prefix = `${encodeURIComponent(name)}=`;
    const item = document.cookie.split('; ').find(value => value.startsWith(prefix));
    return item ? decodeURIComponent(item.slice(prefix.length)) : '';
}

async function apiFetch(input, options = {}) {
    const request = { ...options, credentials: 'include' };
    const method = String(request.method || 'GET').toUpperCase();
    if (!['GET', 'HEAD', 'OPTIONS'].includes(method)) {
        const headers = new Headers(request.headers || {});
        const csrf = readCookie('ai_empowerment_csrf');
        if (csrf) headers.set('X-CSRF-Token', csrf);
        request.headers = headers;
    }
    const response = await nativeFetch(input, request);
    if (
        response.status === 401
        && !String(input).includes('/auth/login')
        && !String(input).includes('/auth/me')
    ) {
        window.dispatchEvent(new CustomEvent('auth-required'));
    }
    return response;
}

const { createApp, ref, nextTick, onMounted, onBeforeUnmount, computed, watch } = Vue;

createApp({
    setup() {
        // ── 状态管理 ──
        const authReady = ref(false);
        const authConfig = ref({ demo_simple_auth: false, password_rule: 'invitation_strong_password' });
        const currentUser = ref(null);
        const loginEmail = ref('');
        const loginPassword = ref('');
        const loginLoading = ref(false);
        const loginError = ref('');
        const activationToken = ref(new URLSearchParams(window.location.search).get('activate') || '');
        const activationPassword = ref('');
        const activationConfirm = ref('');
        const activationLoading = ref(false);
        const activationMessage = ref('');
        const projectMembers = ref([]);
        const newMemberName = ref('');
        const newMemberEmail = ref('');
        const newMemberRole = ref('project_member');
        const newMemberDuty = ref('');
        const memberSaving = ref(false);
        const memberMsg = ref('');
        const latestActivationUrl = ref('');
        const userInitial = computed(() => (currentUser.value?.display_name || currentUser.value?.email || '用').slice(0, 1));
        const userPermissions = computed(() => new Set(currentUser.value?.permissions || []));
        const can = permission => userPermissions.value.has('*') || userPermissions.value.has(permission);
        const tabPermissions = {
            chat: 'knowledge.read', knowledge: 'knowledge.read', graph: 'knowledge.read',
            onboarding: 'onboarding.read', tasks: 'task.read', jobs: 'job.read', feishu: 'feishu.read', resignation: 'handover.read',
            settings: 'settings.manage',
        };
        const tab = ref('chat');
        const mobileNavOpen = ref(false);
        const messages = ref([]);
        const input = ref('');
        const loading = ref(false);
        const useRag = ref(true);
        const chatMode = ref('smart');
        // 演示期间暂不开放 RAG 回归评测入口，保留功能与已有数据以便恢复。
        const ragEvaluationEnabled = false;
        const ragEvaluationOpen = ref(false);
        const ragEvaluationSets = ref([]);
        const selectedRagEvaluationSet = ref(null);
        const ragEvaluationLoading = ref(false);
        const ragEvaluationMessage = ref('');
        const ragEvaluationSetDraft = ref({ name: '', version: 1, description: '' });
        const ragEvaluationCaseDraft = ref({
            question: '', scenario: 'fact', expectedPoints: '', allowedSources: '', expectNoAnswer: false,
        });
        const documents = ref([]);
        const stats = ref({ total_chunks: 0, knowledge_graph: { nodes: 0, edges: 0 }, project_name: '默认项目' });
        const messagesRef = ref(null);
        const llmReady = ref(false);
        const storageReady = ref(false);
        const baseUrl = ref(APP_ORIGIN);
        const activeRetrievalStep = ref(0);
        const activeIngestionStep = ref(3);
        const activeHandoverStep = ref(2);

        async function loadAuthConfig() {
            try {
                const res = await nativeFetch(`${API_BASE}/auth/config`, { credentials: 'include' });
                if (res.ok) authConfig.value = await res.json();
            } catch (error) {
                console.error('认证模式加载失败:', error);
            }
            if (authConfig.value.demo_simple_auth && activationToken.value) {
                activationToken.value = '';
                history.replaceState({}, '', window.location.pathname);
            }
        }

        async function loadCurrentUser() {
            try {
                const res = await apiFetch(`${API_BASE}/auth/me`);
                if (!res.ok) {
                    currentUser.value = null;
                    return false;
                }
                const data = await res.json();
                currentUser.value = data.user || null;
                uploaderName.value = currentUser.value?.display_name || '';
                return Boolean(currentUser.value);
            } catch (error) {
                currentUser.value = null;
                return false;
            } finally {
                authReady.value = true;
            }
        }

        async function login() {
            loginLoading.value = true;
            loginError.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/auth/login`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email: loginEmail.value.trim(), password: loginPassword.value }),
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok) throw new Error(data.detail || '登录失败');
                currentUser.value = data.user;
                uploaderName.value = data.user?.display_name || '';
                loginPassword.value = '';
                await initializeWorkspace();
            } catch (error) {
                loginError.value = error.message || '登录失败';
            } finally {
                loginLoading.value = false;
            }
        }

        async function activateAccount() {
            activationMessage.value = '';
            if (activationPassword.value !== activationConfirm.value) {
                activationMessage.value = '两次输入的密码不一致';
                return;
            }
            activationLoading.value = true;
            try {
                const res = await apiFetch(`${API_BASE}/auth/activate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ token: activationToken.value, password: activationPassword.value }),
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok) throw new Error(data.detail || '激活失败');
                activationMessage.value = '账号激活成功，请使用邮箱和新密码登录';
                loginEmail.value = data.user?.email || loginEmail.value;
                activationToken.value = '';
                activationPassword.value = '';
                activationConfirm.value = '';
                history.replaceState({}, '', window.location.pathname);
            } catch (error) {
                activationMessage.value = error.message || '激活失败';
            } finally {
                activationLoading.value = false;
            }
        }

        async function logout() {
            try { await apiFetch(`${API_BASE}/auth/logout`, { method: 'POST' }); } catch (error) {}
            currentUser.value = null;
            messages.value = [];
            if (feishuEventSource) feishuEventSource.close();
        }

        function handleAuthRequired() {
            currentUser.value = null;
            loginError.value = '登录已失效，请重新登录';
            if (feishuEventSource) feishuEventSource.close();
        }

        async function loadProjectMembers() {
            if (!currentUser.value) return;
            try {
                const res = await apiFetch(`${API_BASE}/projects/current/members`);
                if (res.ok) projectMembers.value = (await res.json()).members || [];
            } catch (error) {
                console.error('成员列表加载失败:', error);
            }
        }

        async function createProjectMember() {
            memberSaving.value = true;
            memberMsg.value = '';
            latestActivationUrl.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/projects/current/members`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        display_name: newMemberName.value.trim(),
                        email: newMemberEmail.value.trim(),
                        role: newMemberRole.value,
                        duty: newMemberDuty.value.trim(),
                    }),
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok) throw new Error(data.detail || '成员创建失败');
                if (data.invitation?.token) {
                    latestActivationUrl.value = `${APP_ORIGIN}/?activate=${encodeURIComponent(data.invitation.token)}`;
                }
                memberMsg.value = data.message || '成员已创建';
                newMemberName.value = '';
                newMemberEmail.value = '';
                newMemberDuty.value = '';
                await loadProjectMembers();
            } catch (error) {
                memberMsg.value = error.message || '成员创建失败';
            } finally {
                memberSaving.value = false;
            }
        }

        async function updateProjectMember(member, changes) {
            memberMsg.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/projects/current/members/${encodeURIComponent(member.user_id)}`, {
                    method: 'PATCH', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(changes),
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok) throw new Error(data.detail || '成员更新失败');
                memberMsg.value = '成员权限已更新';
                await loadProjectMembers();
            } catch (error) {
                memberMsg.value = error.message || '成员更新失败';
                await loadProjectMembers();
            }
        }

        async function copyActivationUrl() {
            if (!latestActivationUrl.value) return;
            await navigator.clipboard.writeText(latestActivationUrl.value);
            memberMsg.value = '激活链接已复制，请通过安全渠道发送给成员';
        }

        // 系统设置状态
        const apiKey = ref('');
        const apiKeyConfigured = ref(false);
        const apiBaseUrl = ref('https://api.openai.com/v1');
        const llmModel = ref('gpt-4o-mini');
        const llmProvider = ref('openai');
        const savingKey = ref(false);
        const testingLLM = ref(false);
        const saveKeyMsg = ref('');
        const llmTestMsg = ref('');
        const llmTestOk = ref(false);
        const llmTestLatency = ref(null);

        // 提供商预设
        const providerPresets = {
            openai:   { baseUrl: 'https://api.openai.com/v1',         model: 'gpt-4o-mini' },
            minimax:  { baseUrl: 'https://api.minimaxi.com/v1',       model: 'MiniMax-M3' },
            deepseek: { baseUrl: 'https://api.deepseek.com',          model: 'deepseek-v4-flash' },
            custom:   { baseUrl: 'https://api.openai.com/v1',         model: 'gpt-4o-mini' },
        };
        const providerLabel = computed(() => {
            const map = { openai: 'OpenAI', minimax: 'MiniMax', deepseek: 'DeepSeek', custom: '自定义' };
            return map[llmProvider.value] || 'LLM';
        });
        function onProviderChange() {
            const p = providerPresets[llmProvider.value];
            if (p) {
                apiBaseUrl.value = p.baseUrl;
                llmModel.value = p.model;
            }
        }
        const providerOptions = [
            { value: 'openai', label: 'OpenAI', desc: '通用模型接口', model: 'gpt-4o-mini' },
            { value: 'minimax', label: 'MiniMax', desc: '国内模型服务', model: 'MiniMax-M3' },
            { value: 'deepseek', label: 'DeepSeek', desc: 'DeepSeek 官方接口', model: 'deepseek-v4-flash' },
            { value: 'custom', label: '自定义兼容接口', desc: '兼容 OpenAI 协议', model: '自定义模型' },
        ];
        function selectProvider(provider) {
            llmProvider.value = provider;
            onProviderChange();
        }

        // 网络代理
        const proxyEnabled = ref(false);
        const proxyUrl = ref('');
        const verifySsl = ref(true);
        const savingNetwork = ref(false);
        const networkMsg = ref('');

        const resettingKnowledgeLibrary = ref(false);
        const knowledgeLibraryResetMessage = ref('');

        // 数据治理
        const governancePolicy = ref({
            raw_message_retention_days: 90,
            access_audit_retention_days: 365,
            processing_history_retention_days: 90,
            backup_retention_count: 10,
            default_sensitivity: 'internal',
            daily_backup_enabled: true,
            daily_retention_enabled: false,
            schedule_time: '02:00',
            timezone: 'Asia/Shanghai',
        });
        const governancePreview = ref({});
        const governanceBackups = ref([]);
        const governanceAudit = ref([]);
        const governanceAuditTotal = ref(0);
        const governanceAuditAction = ref('');
        const governanceAuditKeyword = ref('');
        const governanceConfirm = ref('');
        const governanceTarget = ref({ target_type: 'asset', target_id: '', confirmation: '' });
        const governanceTargetPreview = ref(null);
        const governanceSchedule = ref({ next_scheduled_at: '', latest_intents: [] });
        const governanceLoading = ref(false);
        const governanceActionLoading = ref('');
        const governanceMessage = ref('');

        // 运行可观测性
        const operationsWindow = ref('24h');
        const operationsOverview = ref({
            metrics: [], components: [], alerts: [], latest_capacity: null,
            latest_recovery_drill: null, maintenance_windows: [], methodology: {},
        });
        const operationsLoading = ref(false);
        const operationsActionLoading = ref(false);
        const operationsMessage = ref('');
        const maintenanceDraft = ref({ title: '', reason: '', starts_at: '', ends_at: '' });

        // Tunnel 隧道管理
        const tunnelRunning = ref(false);
        const tunnelUrl = ref('');
        const tunnelLoading = ref(false);
        const tunnelMsg = ref('');

        // 飞书状态
        const feishuConfigured = ref(false);
        const feishuLoading = ref(false);
        const feishuMsg = ref('');
        const pushChatId = ref('');
        const feishuAppId = ref('');
        const feishuAppSecret = ref('');
        const feishuDefaultChatId = ref('');
        const feishuWorkspace = ref({ connection: {}, groups: [], messages: [], candidates: [], assets: [], audit: [], summary: {}, diagnostics: {} });
        const feishuView = ref('overview');
        const feishuSelectedGroupId = ref('');
        const feishuContentGroupId = ref('');
        const feishuHistoryQuery = ref('');
        const feishuContentStatus = ref('');
        const feishuMessageType = ref('');
        const feishuContentSection = ref('assets');
        const feishuHistoryType = ref('assets');
        const feishuDateRange = ref('7');
        const feishuContentMessages = ref([]);
        const feishuContentAssets = ref([]);
        const feishuContentLoading = ref(false);
        const feishuContentPage = ref(1);
        const feishuContentPageSize = 20;
        const feishuContentTotal = ref(0);
        const feishuContentPages = ref(1);
        const feishuWorkspaceLoading = ref(false);
        const feishuSyncing = ref(false);
        const feishuSavingGroup = ref(false);
        const feishuReviewingId = ref('');
        const feishuEnrichingId = ref('');
        const feishuShowCredentialForm = ref(false);
        const feishuLiveSyncState = ref('connecting');
        const feishuDiscoveredChats = ref([]);
        const feishuDiscoveringChats = ref(false);
        const feishuDiscoveryMsg = ref('');
        const feishuSelectedCandidateIds = ref([]);
        const feishuSelectedAssetIds = ref([]);
        const feishuBatchCategory = ref('proj_communication');
        const feishuBatchLoading = ref(false);
        const feishuRevokeConfirmOpen = ref(false);
        const feishuRevokeReason = ref('');
        const feishuRevokeTargets = ref([]);
        const feishuPolicyPreview = ref(null);
        const feishuPolicyPreviewLoading = ref(false);
        const feishuPermissionResult = ref(null);
        const feishuPermissionLoading = ref(false);
        const feishuUndoAsset = ref(null);
        let feishuEventSource = null;
        let feishuContentRequestId = 0;
        let feishuContentReloadTimer = null;
        const feishuGroupDraft = ref({
            chat_id: '', name: '', description: '', avatar: '', chat_type: 'group',
            collection_mode: 'auto', default_category: 'proj_communication', retention_days: 90, bot_enabled: true,
            external: false, confidence_threshold: 0.85, aggregation_window_minutes: 30,
        });

        // ── 知识库状态 ──
        const categories = ref([]);
        const showUploadModal = ref(false);
        const showCatModal = ref(false);
        const uploaderName = ref('');
        const uploadCategory = ref('');
        const uploadFileSelected = ref(false);
        const uploadFileName = ref('');
        const uploadFileData = ref(null);
        const uploadFilesData = ref([]);
        const uploadMode = ref('create');
        const uploadTargetAssetId = ref('');
        const uploadInputKey = ref(0);
        const kbUploading = ref(false);
        const kbUploadMsg = ref('');
        // A task attachment is always a new, independently auditable knowledge asset.
        const knowledgeTaskUploadContext = ref(null);
        const lastIngestion = ref(null);
        const graphImportHint = ref('');
        const groupedDocs = ref({});
        const kbLoading = ref(false);
        const kbLoadError = ref('');
        const previewData = ref(null);
        const previewFilename = ref('');
        const knowledgeAssets = ref([]);
        const knowledgeAssetSummary = ref({});
        const knowledgeAssetsLoading = ref(false);
        const knowledgeAssetsError = ref('');
        const knowledgeAssetsAccessDenied = ref(false);
        const knowledgeSearch = ref('');
        const knowledgeStatusFilter = ref('');
        const knowledgeCategoryFilter = ref('');
        const selectedKnowledgeAsset = ref(null);
        const knowledgeAssetVersions = ref([]);
        const knowledgeAssetDetailLoading = ref(false);
        const knowledgeAssetDetailError = ref('');
        const knowledgeAssetDetailAccessDenied = ref(false);
        const knowledgeAssetDetailUnavailable = ref('');
        const knowledgeAssetSaving = ref(false);
        const knowledgeAssetMessage = ref('');
        const knowledgeRevokeReason = ref('');
        const knowledgeRevokeConfirmOpen = ref(false);
        const knowledgeVersionPreviewingId = ref('');
        const knowledgeAssetDraft = ref({ title: '', owner_user_id: '', review_due_at: '', visibility: 'project' });

        // 分类管理
        const newCatName = ref('');
        const editingCatIdx = ref(-1);
        const editingCatValue = ref('');

        const categoryLabels = {
            eef_culture: '组织文化与结构',
            eef_standard: '行业标准与法规',
            eef_market: '市场环境',
            eef_infrastructure: '基础设施',
            opa_process: '流程与规范',
            opa_template: '模板与工具',
            opa_lessons: '经验教训库',
            opa_training: '培训材料',
            opa_historical: '历史数据',
            proj_plan: '项目管理计划',
            proj_tailoring: '裁剪指南',
            proj_decision: '决策记录',
            proj_risk: '风险日志',
            proj_report: '报告与状态',
            proj_requirement: '需求文档',
            proj_architecture: '技术架构',
            proj_communication: '沟通记录',
            proj_acceptance: '验收标准',
        };

        function categoryLabel(category) {
            return categoryLabels[category] || category || '未分类';
        }

        function isBuiltInCategory(category) {
            return Object.prototype.hasOwnProperty.call(categoryLabels, category);
        }

        const knowledgeStatusMeta = {
            draft: { label: '草稿', tone: 'neutral' },
            pending_review: { label: '待审核', tone: 'warning' },
            active: { label: '已生效', tone: 'success' },
            review_due: { label: '待复审', tone: 'warning' },
            expired: { label: '已失效', tone: 'neutral' },
            revoked: { label: '已撤销', tone: 'danger' },
            deleted: { label: '已删除', tone: 'neutral' },
        };
        const knowledgeStatusOptions = [
            { value: '', label: '全部状态' },
            ...Object.entries(knowledgeStatusMeta)
                .filter(([value]) => value !== 'deleted')
                .map(([value, meta]) => ({ value, label: meta.label })),
        ];
        const knowledgeVisibilityMeta = {
            project: { label: '项目可见', detail: '项目成员可访问' },
            organization: { label: '组织可见', detail: '组织成员可访问' },
            restricted: { label: '指定成员', detail: '按访问策略授权' },
            private: { label: '仅责任人', detail: '仅责任人与管理员可访问' },
        };

        function knowledgeAssetId(asset) {
            return String(asset?.asset_id || asset?.id || '');
        }

        function knowledgeAssetTitle(asset) {
            return asset?.title || asset?.name || asset?.source_file || asset?.filename || '未命名知识资产';
        }

        function knowledgeAssetOptionLabel(asset) {
            const title = knowledgeAssetTitle(asset);
            const category = categoryLabel(asset?.category || asset?.knowledge_category || '');
            const assetId = knowledgeAssetId(asset);
            const suffix = assetId ? `#${assetId.slice(-6)}` : '编号待生成';
            return `${title} · ${category} · ${suffix}`;
        }

        function knowledgeVersionDocuments(version) {
            return Array.isArray(version?.documents) ? version.documents.filter(Boolean) : [];
        }

        function knowledgeCurrentVersion(asset) {
            if (asset?.current_version && typeof asset.current_version === 'object') {
                return {
                    ...asset.current_version,
                    projections: asset.current_version.projections || asset.projections,
                    source: asset.current_version.source || asset.source,
                };
            }
            const assetId = knowledgeAssetId(asset);
            const currentId = String(asset?.current_version_id || '');
            const assetVersions = knowledgeAssetVersions.value.filter(version => !version?.asset_id || String(version.asset_id) === assetId);
            return assetVersions.find(version => currentId && String(version?.version_id || '') === currentId)
                || (assetId === knowledgeAssetId(selectedKnowledgeAsset.value)
                    ? assetVersions.find(version => String(version?.status || '').toLowerCase() === 'active')
                    : null)
                || null;
        }

        function knowledgeAssetSourceRecord(asset) {
            const source = asset?.source && typeof asset.source === 'object' ? asset.source : null;
            if (source) return source;
            if (typeof asset?.source === 'string' && asset.source) return { source_file: asset.source };
            if (asset?.metadata?.source_file || asset?.metadata?.stored_file || asset?.metadata?.filename) return asset.metadata;
            const version = knowledgeCurrentVersion(asset);
            if (version?.source && typeof version.source === 'object') return version.source;
            return knowledgeVersionDocuments(version)[0] || null;
        }

        function knowledgeAssetSource(asset) {
            const source = knowledgeAssetSourceRecord(asset);
            return source?.source_file || source?.display_name || source?.external_key || source?.name || source?.filename || asset?.source_file || asset?.source_name
                || asset?.filename || asset?.current_version?.source_file || '来源待补充';
        }

        function knowledgeAssetStatus(asset) {
            const raw = String(asset?.status || asset?.asset_status || 'draft').toLowerCase();
            return {
                published: 'active', effective: 'active', enabled: 'active',
                pending: 'pending_review', reviewing: 'pending_review',
                due: 'review_due', invalid: 'expired', inactive: 'expired',
            }[raw] || raw;
        }

        function knowledgeStatusLabel(assetOrStatus) {
            const status = typeof assetOrStatus === 'string' ? assetOrStatus : knowledgeAssetStatus(assetOrStatus);
            return knowledgeStatusMeta[status]?.label || status || '状态未知';
        }

        function knowledgeStatusTone(assetOrStatus) {
            const status = typeof assetOrStatus === 'string' ? assetOrStatus : knowledgeAssetStatus(assetOrStatus);
            return knowledgeStatusMeta[status]?.tone || 'neutral';
        }

        function knowledgeOwnerName(asset) {
            const ownerId = String(asset?.owner_user_id || asset?.owner?.user_id || '');
            const member = projectMembers.value.find(item => String(item?.user_id || '') === ownerId);
            return asset?.owner?.display_name || asset?.owner_name || asset?.owner_display_name || asset?.responsible_name
                || member?.display_name || (ownerId ? ownerId : '待分配');
        }

        function knowledgeVersionLabel(asset) {
            const currentVersion = knowledgeCurrentVersion(asset);
            const version = currentVersion?.version_no ?? asset?.current_version?.version_no ?? asset?.current_version_number
                ?? asset?.version_no ?? asset?.version_number;
            if (version !== undefined && version !== null && version !== '') return `v${String(version).replace(/^v/i, '')}`;
            return asset?.current_version_label || (asset?.current_version_id ? '当前版本' : '版本待生成');
        }

        function knowledgeReviewDate(asset) {
            return asset?.review_due_at || asset?.review_at || asset?.next_review_at || asset?.current_version?.review_due_at || '';
        }

        function formatKnowledgeDate(value, withTime = false) {
            if (!value) return '';
            const date = new Date(value);
            if (Number.isNaN(date.getTime())) return String(value).slice(0, withTime ? 16 : 10);
            return new Intl.DateTimeFormat('zh-CN', {
                year: 'numeric', month: '2-digit', day: '2-digit',
                ...(withTime ? { hour: '2-digit', minute: '2-digit', hour12: false } : {}),
            }).format(date).replaceAll('/', '-');
        }

        function knowledgeReviewMeta(asset) {
            const value = knowledgeReviewDate(asset);
            if (!value) return { label: '复审待设置', tone: 'muted' };
            const timestamp = new Date(value).getTime();
            const overdue = Number.isFinite(timestamp) && timestamp < Date.now() && !['expired', 'revoked'].includes(knowledgeAssetStatus(asset));
            return { label: `${formatKnowledgeDate(value)}${overdue ? ' 已逾期' : ' 复审'}`, tone: overdue ? 'danger' : 'muted' };
        }

        function knowledgeVisibility(asset) {
            return String(asset?.visibility || asset?.access_scope || asset?.permission_scope || 'project');
        }

        function knowledgeVisibilityLabel(asset) {
            return knowledgeVisibilityMeta[knowledgeVisibility(asset)]?.label || '权限待确认';
        }

        function knowledgeVisibilityDetail(asset) {
            return asset?.access_policy_name || knowledgeVisibilityMeta[knowledgeVisibility(asset)]?.detail || '来源权限待确认';
        }

        function knowledgeSourceVisibility(asset) {
            const source = knowledgeAssetSourceRecord(asset);
            return String(source?.visibility || asset?.visibility || 'project');
        }

        function knowledgeVisibilityOptions(asset) {
            const sourceVisibility = knowledgeSourceVisibility(asset);
            const allowed = {
                organization: ['organization', 'project', 'private'],
                project: ['project', 'private'],
                restricted: ['restricted', 'private'],
                private: ['private'],
            }[sourceVisibility] || ['project', 'private'];
            const current = knowledgeVisibility(asset);
            if (current === 'restricted' && !allowed.includes('restricted')) allowed.splice(Math.max(0, allowed.length - 1), 0, 'restricted');
            return allowed.map(value => ({ value, ...knowledgeVisibilityMeta[value] }));
        }

        function knowledgeAssetCategories(asset) {
            const values = Array.isArray(asset?.categories) ? asset.categories : [asset?.category || asset?.primary_category].filter(Boolean);
            return values.map(item => typeof item === 'string' ? item : item?.name || item?.category || item?.key).filter(Boolean);
        }

        function knowledgeAssetCategoryLabel(asset) {
            const values = knowledgeAssetCategories(asset);
            return values.length ? values.map(categoryLabel).join('、') : '未分类';
        }

        function knowledgeProjectionItems(version) {
            const labels = { vector: '向量检索', graph: '知识图谱', context: '项目语境', readiness: '新人赋能' };
            return Object.entries(version?.projections || {}).map(([key, value]) => ({
                key,
                label: labels[key] || key,
                status: String(value?.status || value || 'pending').toLowerCase(),
                error: value?.last_error || value?.error || '',
            }));
        }

        function knowledgeProjectionLabel(status) {
            return {
                ready: '已就绪', building: '构建中', pending: '等待处理', repair_required: '需要修复',
                obsolete: '已停用', failed: '处理失败',
            }[String(status || '').toLowerCase()] || status || '状态未知';
        }

        function knowledgeProcessingStatus(asset) {
            const direct = String(asset?.processing_status || '').toLowerCase();
            if (direct) return direct;
            const version = knowledgeCurrentVersion(asset) || asset?.current_version;
            const projections = knowledgeProjectionItems(version);
            if (String(version?.status || '').toLowerCase() === 'failed' || projections.some(item => ['failed', 'repair_required'].includes(item.status))) return 'failed';
            if (['preparing', 'processing'].includes(String(version?.status || '').toLowerCase()) || projections.some(item => ['pending', 'building'].includes(item.status))) return 'processing';
            return 'completed';
        }

        function knowledgeMessageIsError(message) {
            return /失败|必须|没有|尚未|不可|无权|不存在|部分同步|仍需处理/.test(String(message || ''));
        }

        function statusNeedsRepair(value) {
            if (value === false) return true;
            if (value === true || value === null || value === undefined || value === '') return false;
            if (typeof value === 'object') {
                if (Array.isArray(value)) return value.some(statusNeedsRepair);
                if ('status' in value) return statusNeedsRepair(value.status);
                return Object.values(value).some(statusNeedsRepair);
            }
            const status = String(value).trim().toLowerCase();
            return [
                'failed', 'error', 'repair_required', 'pending_refresh', 'partial_success', 'partial',
                'warning', 'degraded', 'pending', 'building', 'processing', 'unsynced',
            ].includes(status);
        }

        function ingestionSyncWarnings(data) {
            const warnings = [];
            if (data?.graph_synced === false) warnings.push('知识图谱');
            if (data?.context_synced === false) warnings.push('项目语境');
            if (data?.readiness_synced === false) warnings.push('新人赋能');

            const currentVersion = data?.asset?.current_version || data?.asset?.version || {};
            const projections = currentVersion?.projections || data?.asset?.projections || {};
            if (statusNeedsRepair(data?.readiness) || statusNeedsRepair(projections?.readiness)) warnings.push('新人赋能');

            const projectionStatus = data?.projection_status
                ?? data?.asset?.projection_status
                ?? currentVersion?.projection_status;
            if (statusNeedsRepair(projectionStatus)) warnings.push('派生投影');

            const labels = { vector: 'RAG 检索', graph: '知识图谱', context: '项目语境', readiness: '新人赋能' };
            Object.entries(projections || {}).forEach(([key, value]) => {
                if (statusNeedsRepair(value)) warnings.push(labels[key] || key);
            });
            return [...new Set(warnings)];
        }

        function knowledgeSummaryValue(key) {
            const aliases = {
                active: ['active', 'effective', 'published', 'active_count'],
                pending_review: ['pending_review', 'pending', 'pending_count', 'review_count'],
                review_due: ['review_due', 'due', 'review_due_count', 'overdue_count'],
                failed: ['failed', 'failed_count', 'processing_failed'],
            };
            for (const name of aliases[key] || [key]) {
                if (knowledgeAssetSummary.value?.[name] !== undefined) return Number(knowledgeAssetSummary.value[name]) || 0;
            }
            if (key === 'failed') return knowledgeAssets.value.filter(asset => knowledgeProcessingStatus(asset) === 'failed').length;
            return knowledgeAssets.value.filter(asset => knowledgeAssetStatus(asset) === key).length;
        }

        const knowledgeActionMetrics = computed(() => [
            { key: 'active', label: '已生效', value: knowledgeSummaryValue('active'), hint: '当前可用于问答' },
            { key: 'pending_review', label: '待审核', value: knowledgeSummaryValue('pending_review'), hint: '等待治理确认' },
            { key: 'review_due', label: '待复审', value: knowledgeSummaryValue('review_due'), hint: '含已到期资产' },
            { key: 'failed', label: '处理失败', value: knowledgeSummaryValue('failed'), hint: '需要重试处理' },
        ]);

        const filteredKnowledgeAssets = computed(() => {
            const query = knowledgeSearch.value.trim().toLowerCase();
            return knowledgeAssets.value.filter((asset) => {
                const statusMatch = !knowledgeStatusFilter.value
                    || (knowledgeStatusFilter.value === 'failed'
                        ? knowledgeProcessingStatus(asset) === 'failed'
                        : knowledgeAssetStatus(asset) === knowledgeStatusFilter.value);
                const categoryMatch = !knowledgeCategoryFilter.value
                    || knowledgeAssetCategories(asset).includes(knowledgeCategoryFilter.value);
                const haystack = [knowledgeAssetTitle(asset), knowledgeAssetSource(asset), knowledgeOwnerName(asset), knowledgeAssetCategoryLabel(asset)]
                    .join(' ').toLowerCase();
                return statusMatch && categoryMatch && (!query || haystack.includes(query));
            });
        });

        function setKnowledgeStatusFilter(status) {
            knowledgeStatusFilter.value = knowledgeStatusFilter.value === status ? '' : status;
        }

        // ── 知识图谱状态 ──
        const graphData = ref({ nodes: [], edges: [] });
        const graphError = ref(false);
        const graphBuilding = ref(false);
        const graphCanvasRef = ref(null);
        const graphSearch = ref('');
        const graphGroupFilter = ref('全部');
        const graphLabelMode = ref('smart');
        const graphZoomLabel = ref('100%');
        const graphSaving = ref(false);
        const graphMsg = ref('');
        const selectedGraphNode = ref(null);
        const selectedGraphEdge = ref(null);
        const graphEditorMode = ref('node');
        const nodeDraft = ref({ label: '', group: '项目知识', type: '知识节点', title: '', size: 18 });
        const newNodeDraft = ref({ label: '', group: '项目知识', type: '知识节点', title: '' });
        const edgeDraft = ref({ source: '', target: '', label: '关联', width: 2, confidence: 0.9, evidence: '', protected: true });
        let graphSimulation = null;
        let graphSaveTimer = null;
        let graphFitToContent = null;

        // ── 离职交接状态 ──
        const resignName = ref('');
        const resignRecipient = ref('');
        const resignDueDate = ref('');
        const resignFiles = ref([]);
        const resignSubmitting = ref(false);
        const resignResult = ref(null);
        const resignError = ref('');
        const handoverRecords = ref([]);
        const handoverLoading = ref(false);
        const handoverLoadError = ref('');
        const acceptingHandoverId = ref('');
        const cancellingHandoverId = ref('');
        const handoverActionMessage = ref('');
        const showHandoverModal = ref(false);
        const selectedHandover = ref(null);
        const handoverDetailLoading = ref(false);
        const handoverActionLoading = ref('');
        const handoverItemDrafts = ref({});
        const handoverItemAssetRefs = ref({});
        const handoverItemAssetOptions = ref({});
        const handoverItemAssetPickers = ref({});
        const handoverItemDraftSaving = ref({});
        const handoverItemDraftMessages = ref({});
        const handoverRejectDrafts = ref({});
        const handoverProxyReasonDrafts = ref({});
        const handoverInventoryExcludeDrafts = ref({});
        const handoverRiskActionDrafts = ref({});
        const handoverRiskDraft = ref({
            title: '', impact: '', severity: 'medium', owner_user_id: '', due_at: '', mitigation: '',
        });
        // 展示层控制：分组折叠、已闭环折叠、按需弹出的输入与表单
        const handoverSectionOpen = ref({ inventory: true, items: true, risks: true });
        const handoverShowClosed = ref({ inventory: false, items: false, risks: false });
        const handoverInventoryExcludeActive = ref({});
        const handoverShowRiskForm = ref(false);
        const handoverItemDraftTimers = new Map();

        // ── 新人赋能状态 ──
        const onboardingRole = ref('developer');
        const onboardingTemplates = ref([]);
        const onboardingTargetUserId = ref('');
        const onboardingData = ref(null);
        const onboardingLoading = ref(false);
        const onboardingError = ref('');
        const supervisedOnboardingPlans = ref([]);
        const supervisedOnboardingPlansLoading = ref(false);
        const supervisedOnboardingPlansError = ref('');
        const onboardingPlanDetailId = ref('');
        const onboardingRoles = computed(() => onboardingTemplates.value.map(template => ({
            value: template.role_key, label: template.name,
        })));
        // 成员职能以岗位模板名称为选项，保证与离职交接的岗位解析一致。
        const dutyOptions = computed(() => onboardingTemplates.value.map(template => ({
            value: template.name, label: template.name,
        })));
        const onboardingRoleLabel = computed(() => onboardingRoles.value.find(
            role => role.value === onboardingRole.value,
        )?.label || onboardingData.value?.role || '待选择');
        const onboardingTargetMembers = computed(() => projectMembers.value.filter(
            member => member.status === 'active' && member.membership_status === 'active',
        ));
        const supervisedOnboardingSummary = computed(() => {
            const plans = supervisedOnboardingPlans.value;
            return {
                planCount: plans.length,
                memberCount: new Set(plans.map(plan => plan.user_id)).size,
                confirmationReadyCount: plans.filter(plan => plan.manager_confirmation?.can_confirm).length,
                blockedPlanCount: plans.filter(plan => Number(plan.blocked_count || 0) > 0).length,
            };
        });
        const onboardingPlanDetailOpen = computed(() => (
            !can('onboarding.manage') || Boolean(onboardingPlanDetailId.value)
        ));
        const onboardingActionLoading = ref('');
        const onboardingEvidenceDrafts = ref({});
        const showOnboardingTemplateModal = ref(false);
        const onboardingTemplateDraft = ref(null);
        const onboardingTaskMessage = ref('');

        // ── 知识任务状态 ──
        const knowledgeTasks = ref([]);
        const knowledgeTaskSummary = ref({});
        const knowledgeTasksLoading = ref(false);
        const knowledgeTasksError = ref('');
        const selectedKnowledgeTask = ref(null);
        const knowledgeTaskDetailLoading = ref(false);
        const knowledgeTaskStatusFilter = ref('active');
        const knowledgeTaskTypeFilter = ref('');
        const knowledgeTaskKeyword = ref('');
        const knowledgeTaskMessage = ref('');
        const knowledgeTaskActionLoading = ref('');
        const knowledgeTaskEvidence = ref('');
        const knowledgeTaskEvidenceUploadMessage = ref('');
        const knowledgeTaskNote = ref('');
        const knowledgeTaskAssignee = ref('');
        const knowledgeTaskChatId = ref('');
        const showKnowledgeTaskCancelModal = ref(false);
        const knowledgeTaskCancelReason = ref('');
        const knowledgeTaskCancelMessage = ref('');
        const showKnowledgeTaskPolicyModal = ref(false);
        const knowledgeTaskPolicyLoading = ref(false);
        const knowledgeTaskPolicySaving = ref(false);
        const knowledgeTaskPolicyScanning = ref(false);
        const knowledgeTaskPolicyMessage = ref('');
        const knowledgeTaskNotificationPolicy = ref({
            enabled: false, target_chat_id: '', timezone: 'Asia/Shanghai',
            quiet_start: '22:00', quiet_end: '08:00', remind_before_hours: 24,
            reminder_interval_hours: 24, escalation_after_hours: 24,
            escalation_interval_hours: 24,
        });
        const showKnowledgeTaskModal = ref(false);
        const knowledgeTaskSaving = ref(false);
        const knowledgeTaskDraft = ref({
            title: '', description: '', task_type: 'manual', priority: 'medium',
            assignee_user_id: '', due_at: '', source_type: 'manual', source_key: '',
            dedupe_key: '', handover_id: '',
        });
        const knowledgeTaskStatusMeta = {
            unassigned: { label: '待分配', tone: 'neutral' },
            open: { label: '待处理', tone: 'warning' },
            in_progress: { label: '处理中', tone: 'info' },
            pending_acceptance: { label: '待验收', tone: 'warning' },
            completed: { label: '已完成', tone: 'success' },
            cancelled: { label: '已取消', tone: 'neutral' },
        };
        const knowledgeTaskTypeMeta = {
            knowledge_gap: '知识缺口', asset_review: '资产复审', asset_publish: '资产发布', feishu_review: '飞书审核',
            handover_gap: '交接补充', qa_feedback: '问答改进', manual: '人工任务',
        };
        const knowledgeTaskCreatableTypeMeta = Object.fromEntries(
            Object.entries(knowledgeTaskTypeMeta).filter(([type]) => type !== 'asset_publish'),
        );
        const knowledgeTaskPriorityMeta = {
            low: '低', medium: '中', high: '高', critical: '紧急',
        };
        const knowledgeTaskEventMeta = {
            created: '创建任务', merged: '合并重复发现', reopened: '再次退回补充',
            claim: '领取任务', assign: '转派负责人',
            start: '开始处理', submit: '提交验收', accept: '验收通过', return: '退回补充',
            cancel: '取消任务', comment: '添加备注', auto_evidence_matched: '自动匹配生效资料',
        };
        const canManageKnowledgeTasks = computed(() => can('task.manage'));
        const knowledgeTaskCan = (action) => (
            selectedKnowledgeTask.value?.allowed_actions || []
        ).includes(action);
        const knowledgeTaskPolicyStatus = computed(() => {
            if (!knowledgeTaskNotificationPolicy.value.enabled) return '自动提醒未开启';
            if (!knowledgeTaskNotificationPolicy.value.effective_target_chat_id
                && !knowledgeTaskNotificationPolicy.value.target_chat_id) return '自动提醒缺少目标群';
            return `自动提醒已开启 · 静默 ${knowledgeTaskNotificationPolicy.value.quiet_start}–${knowledgeTaskNotificationPolicy.value.quiet_end}`;
        });
        const filteredKnowledgeTasks = computed(() => {
            const keyword = knowledgeTaskKeyword.value.trim().toLowerCase();
            return knowledgeTasks.value.filter((task) => {
                const statusMatch = knowledgeTaskStatusFilter.value === 'active'
                    ? !['completed', 'cancelled'].includes(task.status)
                    : !knowledgeTaskStatusFilter.value || task.status === knowledgeTaskStatusFilter.value;
                const typeMatch = !knowledgeTaskTypeFilter.value || task.task_type === knowledgeTaskTypeFilter.value;
                const text = `${task.title || ''} ${task.description || ''} ${task.assignee_name || ''}`.toLowerCase();
                return statusMatch && typeMatch && (!keyword || text.includes(keyword));
            });
        });

        const processingJobs = ref([]);
        const processingJobSummary = ref({});
        const processingJobsLoading = ref(false);
        const processingJobsError = ref('');
        const selectedProcessingJob = ref(null);
        const processingJobDetailLoading = ref(false);
        const processingJobStatusFilter = ref('active');
        const processingJobTypeFilter = ref('');
        const processingJobKeyword = ref('');
        const processingJobPage = ref(1);
        const processingJobPages = ref(1);
        const processingJobTotal = ref(0);
        const processingJobActionLoading = ref('');
        const processingJobMessage = ref('');
        const canManageProcessingJobs = computed(() => can('job.manage'));
        const canRetrySelectedProcessingJob = computed(() => Boolean(
            selectedProcessingJob.value
            && (selectedProcessingJob.value.allowed_actions || []).includes('retry')
        ));
        const processingJobStatusMeta = {
            queued: { label: '排队中', tone: 'neutral' }, running: { label: '处理中', tone: 'info' },
            retry_wait: { label: '等待重试', tone: 'warning' }, succeeded: { label: '已完成', tone: 'success' },
            failed: { label: '处理失败', tone: 'danger' }, cancelled: { label: '已取消', tone: 'neutral' },
        };
        const processingJobTypeMeta = {
            graph_rebuild: '知识图谱重建', projection_repair: '知识投影修复', feishu_group_sync: '飞书历史同步',
            feishu_candidate_batch: '飞书候选批量处理', feishu_message_enrichment: '飞书图片识别',
            database_backup: '数据库备份', data_retention: '数据保留清理', targeted_data_deletion: '定向数据删除', document_ingestion: '文档知识入库',
            capacity_validation: '容量基线验证', recovery_drill: '隔离恢复演练',
            knowledge_task_notification: '知识任务提醒',
        };
        const processingJobStageMeta = {
            queued: '等待执行', starting: '准备处理', graph_build: '构建知识关系', graph_save: '保存图谱',
            projection_scan: '检查投影', projection_save: '确认投影', feishu_download: '读取飞书消息',
            feishu_enrich: '提取消息内容', feishu_archive: '归档并聚合', feishu_publish: '知识入库',
            feishu_batch_validate: '复核批量权限', feishu_batch_item: '逐项处理候选',
            feishu_batch_finalize: '汇总逐项结果',
            feishu_enrich_validate: '复核消息与权限', feishu_enrich_extract: '下载并识别图片',
            feishu_enrich_archive: '保存识别结果', feishu_enrich_publish: '同步知识候选',
            backup_database: '创建一致性备份', backup_verify: '校验备份', backup_complete: '完成备份',
            retention_scan: '核对保留范围', retention_audit: '记录清理结果',
            target_delete_validate: '核对定向范围', target_delete_originals: '清除原件正文',
            target_delete_projections: '清除派生投影', target_delete_complete: '完成定向删除',
            document_validate: '复核上传权限', document_ingest: '解析并发布知识',
            document_parse: '读取文档结构', document_ocr_render: '渲染扫描页面',
            document_ocr_recognize: '识别文档文字',
            capacity_prepare: '准备容量样本', capacity_query: '执行容量查询', capacity_save: '保存容量报告',
            recovery_select: '选择有效备份', recovery_restore: '隔离恢复备份', recovery_record: '保存演练报告',
            notification_validate: '复核提醒策略', notification_deliver: '投递提醒摘要',
            retry_wait: '等待自动重试', completed: '处理完成', failed: '处理失败', cancelled: '已取消',
        };

        // ── 计算属性 ──
        const tabLabels = {
            chat: '知识问答',
            knowledge: '知识库',
            graph: '知识图谱',
            onboarding: '新人赋能',
            tasks: '知识任务',
            jobs: '处理任务',
            feishu: '飞书集成',
            resignation: '离职交接',
            settings: '系统设置',
        };
        const pageTitle = computed(() => tabLabels[tab.value] || '知识问答');
        const currentTabLabel = computed(() => tabLabels[tab.value] || '知识问答');

        const retrievalSteps = ['检索文档', '匹配交接资料', '生成答案', '附加引用'];
        const ingestionSteps = [
            { title: '解析中', desc: 'OCR 识别与格式解析', progress: 45 },
            { title: '切分知识块', desc: '文本切分与清洗', progress: 70 },
            { title: '向量入库', desc: '生成向量并存储', progress: 85 },
            { title: '图谱更新', desc: '关联知识节点与关系', progress: 100 },
        ];
        const handoverFlowSteps = ['人员登记', '文档归档', '指定接替人', '确认接收', '新人可查'];
        const activeHandoverRecord = computed(() => handoverRecords.value[0] || resignResult.value?.handover || null);
        const handoverChecklist = computed(() => {
            const record = activeHandoverRecord.value;
            return [
                { title: '人员与职能信息登记', status: record ? '已完成' : '待登记' },
                { title: '交接文档归档与入库', status: record?.file_count ? '已完成' : '待补充' },
                { title: '指定接替人与完成日期', status: record?.recipient ? '已完成' : '待补充' },
                { title: '接替人确认接收', status: ['accepted', 'completed'].includes(record?.status) ? '已完成' : '待确认' },
                { title: '生成接替人学习计划', status: record?.learning_plan_status === 'ready' ? '已生成' : record?.status === 'accepted' ? '待处理' : '待确认' },
            ];
        });
        const handoverRisks = computed(() => {
            const risks = handoverRecords.value.flatMap((record) => record.risks || []);
            return [...new Set(risks)];
        });
const starterQuestions = [
  '本项目当前有哪些关键技术架构和依赖？',
  '当前交接中有哪些待处理风险？',
  '作为新成员我应先阅读哪些资料？',
];

        const flatGroupedDocs = computed(() => {
            const groups = groupedDocs.value || {};
            return Object.entries(groups).flatMap(([category, docs]) => {
                return (docs || []).map((doc) => ({
                    name: doc.source_file || doc.name || '未命名文档',
                    ext: (doc.source_file || doc.name || 'DOC').split('.').pop().slice(0, 3).toUpperCase(),
                    status: '已入库',
                    chunks: doc.chunks || 0,
                    category,
                    progress: 100,
                }));
            });
        });
        const dashboardDocs = computed(() => flatGroupedDocs.value);

        const displayChunkCount = computed(() => Number(stats.value.total_chunks || 0).toLocaleString());
        const displayGraphNodes = computed(() => Number(stats.value.knowledge_graph?.nodes ?? graphData.value.nodes?.length ?? 0).toLocaleString());
        const displayGraphEdges = computed(() => Number(stats.value.knowledge_graph?.edges ?? graphData.value.edges?.length ?? 0).toLocaleString());
        const displayCitationCount = computed(() => dashboardDocs.value.length);
        const graphPreviewNodes = computed(() => (graphData.value.nodes || []).slice(0, 5));
        const graphPreviewEdges = computed(() => (graphData.value.edges || []).slice(0, 4));
        const feishuPreviewText = computed(() => `当前已入库 ${dashboardDocs.value.length} 份文档，${displayChunkCount.value} 个知识块，${handoverRecords.value.length} 个交接任务，${handoverRisks.value.length} 项待处理风险。`);
        const feishuGroups = computed(() => feishuWorkspace.value.groups || []);
        const feishuMessages = computed(() => feishuWorkspace.value.messages || []);
        const feishuCandidates = computed(() => feishuWorkspace.value.candidates || []);
        const feishuAssets = computed(() => feishuWorkspace.value.assets || []);
        const feishuAudit = computed(() => feishuWorkspace.value.audit || []);
        const feishuProjectGroups = computed(() => feishuGroups.value.filter((group) => group.chat_type !== 'p2p'));
        const feishuSummary = computed(() => feishuWorkspace.value.summary || {});
        const feishuConnection = computed(() => feishuWorkspace.value.connection || {});
        const feishuTransport = computed(() => feishuConnection.value.transport || {});
        const feishuDiagnostics = computed(() => feishuWorkspace.value.diagnostics || {});
        const feishuEventConnected = computed(() => !!feishuTransport.value.connected);
        const feishuSelectedGroup = computed(() => feishuGroups.value.find((group) => group.chat_id === feishuSelectedGroupId.value) || null);
        const feishuAvailableChats = computed(() => {
            const connectedById = new Map(feishuProjectGroups.value.map((group) => [group.chat_id, group]));
            const discoveredIds = new Set(feishuDiscoveredChats.value.map((chat) => chat.chat_id));
            const discovered = feishuDiscoveredChats.value.map((chat) => {
                const group = connectedById.get(chat.chat_id);
                const syncedCount = feishuMessages.value.filter((message) => message.chat_id === chat.chat_id).length;
                return { ...chat, ...(group || {}), name: chat.name || group?.name, connected: !!group, synced_count: syncedCount };
            });
            const manual = feishuProjectGroups.value
                .filter((group) => !discoveredIds.has(group.chat_id))
                .map((group) => ({
                    ...group,
                    connected: true,
                    synced_count: feishuMessages.value.filter((message) => message.chat_id === group.chat_id).length,
                }));
            return [...discovered, ...manual];
        });
        const feishuPendingMessages = computed(() => feishuMessages.value.filter((message) => message.content_status === 'review_required' || message.review_status === 'pending'));
        const feishuPendingCandidates = computed(() => feishuCandidates.value.filter((candidate) => ['review_required', 'failed'].includes(candidate.status)));
        const feishuAllPendingSelected = computed(() => feishuPendingCandidates.value.length > 0 && feishuPendingCandidates.value.every((candidate) => feishuSelectedCandidateIds.value.includes(candidate.candidate_id)));
        const feishuPublishedAssets = computed(() => feishuContentAssets.value.filter((asset) => asset.status === 'published'));
        const feishuAllAssetsSelected = computed(() => feishuPublishedAssets.value.length > 0 && feishuPublishedAssets.value.every((asset) => feishuSelectedAssetIds.value.includes(asset.asset_id)));
        const feishuVisibleMessages = computed(() => feishuContentMessages.value);
        const feishuArchiveCount = computed(() => Math.max(
            0,
            Number(feishuSummary.value.message_count || 0)
                - Number(feishuSummary.value.excluded_count || 0)
                - Number(feishuSummary.value.reverted_message_count || 0),
        ));
        const feishuHistoryCount = computed(() => (
            Number(feishuSummary.value.reverted_count || 0)
            + Number(feishuSummary.value.excluded_count || 0)
        ));
        const feishuContentGroups = computed(() => {
            const groupNames = new Map(feishuGroups.value.map((group) => [group.chat_id, group.name || '飞书会话']));
            const today = new Date();
            const yesterday = new Date(today);
            yesterday.setDate(today.getDate() - 1);
            const dateKey = (value) => {
                const raw = String(value || '').trim();
                let date;
                if (/^\d+$/.test(raw)) {
                    const number = Number(raw);
                    date = new Date(number > 9_999_999_999 ? number : number * 1000);
                } else {
                    date = new Date(raw);
                }
                if (Number.isNaN(date.getTime())) date = new Date(0);
                return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
            };
            const todayKey = dateKey(today.toISOString());
            const yesterdayKey = dateKey(yesterday.toISOString());
            const groups = new Map();
            feishuVisibleMessages.value.forEach((message) => {
                const day = dateKey(message.create_time || message.synced_at);
                const key = `${message.chat_id || 'unknown'}:${day}`;
                if (!groups.has(key)) {
                    groups.set(key, {
                        key,
                        name: groupNames.get(message.chat_id) || '飞书会话',
                        date: day === todayKey ? '今天' : day === yesterdayKey ? '昨天' : day,
                        items: [],
                    });
                }
                groups.get(key).items.push(message);
            });
            return [...groups.values()];
        });
        const feishuPipelineStages = computed(() => {
            const summary = feishuSummary.value;
            const stages = [
                { key: 'received', label: '消息归档', count: Number(summary.message_count || 0), detail: '保留来源与原始消息' },
                { key: 'evaluated', label: '内容评估', count: Number(summary.evaluated_count || 0), detail: '去重并判断知识价值' },
                { key: 'candidates', label: '聚合候选', count: Number(summary.candidate_count || 0), detail: '按主题与讨论窗口聚合' },
                { key: 'published', label: '知识入库', count: Number(summary.published_count || 0), detail: '进入 RAG、图谱与新人路径' },
            ];
            const maximum = Math.max(1, ...stages.map((item) => item.count));
            return stages.map((item) => ({
                ...item,
                width: item.count ? Math.max(8, Math.round(item.count * 100 / maximum)) : 0,
            }));
        });
        const feishuTodayConversion = computed(() => {
            const summary = feishuSummary.value;
            const stages = [
                { key: 'received', label: '今日接收', count: Number(summary.today_received || 0), detail: '授权采集源原始内容' },
                { key: 'valuable', label: '有效内容', count: Number(summary.today_valuable_count || 0), detail: '通过价值与安全判断' },
                { key: 'candidates', label: '形成候选', count: Number(summary.today_candidate_count || 0), detail: '完成主题聚合与去重' },
                { key: 'published', label: '同批入库', count: Number(summary.today_published_count || 0), detail: '今日消息形成的知识' },
            ];
            const maximum = Math.max(1, ...stages.map((item) => item.count));
            return {
                stages: stages.map((item) => ({
                    ...item,
                    width: item.count ? Math.max(8, Math.round(item.count * 100 / maximum)) : 0,
                })),
                conversionRate: Number(summary.today_conversion_rate || 0),
                automationRate: Number(summary.today_automation_rate || 0),
                completedCount: Number(summary.today_completed_count || 0),
            };
        });
        const feishuProcessingComposition = computed(() => {
            const summary = feishuSummary.value;
            const total = Math.max(1, Number(summary.message_count || 0));
            return [
                { key: 'published', label: '参与知识入库', count: Number(summary.published_message_count || 0), tone: 'success' },
                { key: 'archived', label: '仅归档', count: Number(summary.archived_count || 0), tone: 'primary' },
                { key: 'excluded', label: '自动排除', count: Number(summary.excluded_count || 0), tone: 'neutral' },
                { key: 'review', label: '需处理', count: Number(summary.review_required_count || 0), tone: 'warning' },
            ].map((item) => ({
                ...item,
                width: item.count ? Math.max(4, Math.round(item.count * 100 / total)) : 0,
            }));
        });
        const feishuRecentActivity = computed(() => {
            const groupNames = new Map(feishuGroups.value.map((group) => [group.chat_id, group.name || '飞书会话']));
            const timestampOf = (message) => {
                const raw = String(message.create_time || message.synced_at || '').trim();
                if (/^\d+$/.test(raw)) {
                    const value = Number(raw);
                    return value > 9_999_999_999 ? value : value * 1000;
                }
                const parsed = Date.parse(raw);
                return Number.isNaN(parsed) ? 0 : parsed;
            };
            const titleOf = (message) => {
                if (message.content_status === 'published') return '知识已形成并入库';
                if (message.message_type === 'image' && message.extraction_status === 'completed') return '图片 OCR 已完成';
                if (message.message_type === 'post' && message.extraction_status === 'completed') return '富文本已解析';
                if (message.message_type === 'file') return '分享文档已归档';
                if (message.content_status === 'excluded') return '低价值内容已过滤';
                return '群消息已归档';
            };
            const detailOf = (message) => {
                const content = String(message.title || message.file_name || message.content || '').trim();
                if (content.includes('```mermaid')) return '流程图原文已归档，可在内容中心查看完整内容';
                if (content.startsWith('[暂不支持直接预览')) return '旧版内容待重新识别，可在内容中心继续处理';
                const cleaned = content
                    .replace(/```[a-zA-Z]*|```/g, ' ')
                    .replace(/[#*_>`~\[\]]/g, ' ')
                    .replace(/\s+/g, ' ')
                    .trim();
                return (cleaned || '内容已进入自动化流程').slice(0, 42);
            };
            return [...feishuMessages.value]
                .sort((a, b) => timestampOf(b) - timestampOf(a))
                .slice(0, 5)
                .map((message) => ({
                    id: message.message_id,
                    title: titleOf(message),
                    source: groupNames.get(message.chat_id) || '飞书会话',
                    detail: detailOf(message),
                    time: formatFeishuTime(message.create_time || message.synced_at),
                    tone: message.content_status === 'published' ? 'success' : message.content_status === 'review_required' ? 'warning' : 'primary',
                    type: feishuMessageTypeLabel(message.message_type),
                }));
        });
        const feishuCategoryOptions = computed(() => [
            { value: 'proj_communication', label: '沟通记录' },
            { value: 'proj_decision', label: '决策记录' },
            { value: 'proj_risk', label: '风险日志' },
            { value: 'opa_lessons', label: '经验教训库' },
            { value: 'opa_process', label: '流程与规范' },
            ...categories.value.filter((item) => !['proj_communication', 'proj_decision', 'proj_risk', 'opa_lessons', 'opa_process'].includes(item)).map((item) => ({ value: item, label: categoryLabels[item] || item })),
        ]);
        const graphGroups = computed(() => {
            const groups = new Set((graphData.value.nodes || []).map((node) => node.group || node.type || '项目知识'));
            return ['全部', ...groups];
        });
        const graphLegendGroups = computed(() => graphGroups.value.slice(1, 5).map((name, index) => ({
            name,
            color: graphColorAt(index),
        })));
        const filteredGraph = computed(() => {
            const keyword = graphSearch.value.trim().toLowerCase();
            const activeGroup = graphGroupFilter.value;
            const nodes = (graphData.value.nodes || []).filter((node) => {
                const inGroup = activeGroup === '全部' || (node.group || node.type || '项目知识') === activeGroup;
                const haystack = `${node.label || ''} ${node.title || ''} ${node.source || ''} ${(node.tags || []).join(' ')}`.toLowerCase();
                return inGroup && (!keyword || haystack.includes(keyword));
            });
            const nodeIds = new Set(nodes.map((node) => node.id));
            const edges = (graphData.value.edges || []).filter((edge) => {
                const source = edge.source || edge.from;
                const target = edge.target || edge.to;
                return nodeIds.has(source) && nodeIds.has(target);
            });
            return { nodes, edges };
        });
        const graphTableNodes = computed(() => filteredGraph.value.nodes.slice(0, 50));
        const graphTableEdges = computed(() => filteredGraph.value.edges.slice(0, 80));
        function graphColorAt(index) {
            return GRAPH_PALETTE[index % GRAPH_PALETTE.length];
        }

        function navigateTo(nextTab) {
            const permission = tabPermissions[nextTab];
            if (permission && !can(permission)) return;
            tab.value = nextTab;
            mobileNavOpen.value = false;
        }

        function openFeishuView(nextView = 'overview') {
            feishuView.value = nextView;
            navigateTo('feishu');
        }

        function handoverStage(record) {
            if (!record) return 0;
            if (record.status === 'completed') return handoverFlowSteps.length - 1;
            if (record.status === 'accepted' && record.learning_plan_status === 'ready') return handoverFlowSteps.length - 1;
            if (record.status === 'accepted') return 3;
            if (record.recipient) return 3;
            if (record.file_count) return 2;
            return 1;
        }

        function applyHandoverStage(record) {
            activeHandoverStep.value = handoverStage(record);
        }

        function handoverStepComplete(index) {
            const record = activeHandoverRecord.value;
            return [
                Boolean(record),
                Boolean(record?.file_count),
                Boolean(record?.recipient_user_id || record?.recipient),
                ['accepted', 'completed'].includes(record?.status),
                record?.learning_plan_status === 'ready',
            ][index] || false;
        }

        function handoverStepClass(index) {
            const done = handoverStepComplete(index);
            return { done, current: !done && index === activeHandoverStep.value };
        }

        function handoverStepStatusLabel(index) {
            if (handoverStepComplete(index)) return '已完成';
            return index === activeHandoverStep.value ? '进行中' : '待处理';
        }

        const handoverItemStatusMeta = {
            pending: { label: '待提交', tone: 'neutral' },
            submitted: { label: '待验收', tone: 'warning' },
            accepted: { label: '已验收', tone: 'success' },
            rejected: { label: '已退回', tone: 'danger' },
        };
        const handoverRiskStatusMeta = {
            open: { label: '待处理', tone: 'danger' },
            mitigating: { label: '缓解中', tone: 'warning' },
            pending_close: { label: '待关闭验收', tone: 'info' },
            closed: { label: '已关闭', tone: 'success' },
        };
        const handoverRiskSeverityMeta = {
            low: '低', medium: '中', high: '高', blocking: '阻断',
        };

        function handoverStatusLabel(record) {
            if (record?.status === 'completed') return '已关闭并封存';
            if (record?.status === 'accepted' && record?.learning_plan_status === 'ready') return '已确认，计划就绪';
            if (record?.status === 'accepted') return '已确认，计划待处理';
            return '待接替人确认';
        }

        function canAcceptHandover(record) {
            return Boolean(
                record?.id
                && record.recipient_user_id === currentUser.value?.user_id
                && record.status !== 'completed'
                && (record.status !== 'accepted' || record.learning_plan_status !== 'ready')
            );
        }

        function canCancelHandover(record) {
            const initiatorUserId = record?.created_by_user_id || record?.departing_user_id;
            return Boolean(
                record?.id
                && record.status === 'pending_acceptance'
                && initiatorUserId === currentUser.value?.user_id
            );
        }

        function handoverItemStatusLabel(status) {
            return handoverItemStatusMeta[status]?.label || status || '未知';
        }

        function handoverRiskStatusLabel(status) {
            return handoverRiskStatusMeta[status]?.label || status || '未知';
        }

        function handoverMemberName(userId) {
            const member = projectMembers.value.find(item => item.user_id === userId);
            return member?.display_name || member?.email || userId || '未指定';
        }

        function canSubmitHandoverItem(item) {
            const record = selectedHandover.value;
            const actor = currentUser.value?.user_id;
            return can('handover.manage') || [
                item?.owner_user_id, record?.departing_user_id, record?.created_by_user_id,
            ].filter(Boolean).includes(actor);
        }

        function canReviewHandover() {
            return can('handover.manage')
                || selectedHandover.value?.recipient_user_id === currentUser.value?.user_id;
        }

        function canReviewHandoverItem(item) {
            return (item?.review_actions || []).includes('accept');
        }

        function canManageHandoverRisk(risk) {
            return can('handover.manage') || risk?.owner_user_id === currentUser.value?.user_id;
        }

        // 交接清单展示层：角色判定、分组折叠、按重要性排序与待办聚合
        const handoverItemRank = { rejected: 0, submitted: 1, pending: 2, accepted: 3 };
        const handoverRiskStatusRank = { open: 0, mitigating: 1, pending_close: 2, closed: 3 };
        const handoverRiskSeverityRank = { blocking: 0, high: 1, medium: 2, low: 3 };

        function handoverParticipantIds() {
            const record = selectedHandover.value;
            const ids = new Set();
            if (!record) return ids;
            [record.departing_user_id, record.created_by_user_id, record.recipient_user_id]
                .filter(Boolean).forEach(id => ids.add(String(id)));
            (record.items || []).forEach(item => { if (item.owner_user_id) ids.add(String(item.owner_user_id)); });
            return ids;
        }

        function isHandoverParticipant() {
            const actor = currentUser.value?.user_id;
            return Boolean(actor) && handoverParticipantIds().has(String(actor));
        }

        function canRegisterHandoverRisk() {
            return can('handover.manage') || (isHandoverParticipant() && can('handover.accept'));
        }

        const handoverInventorySorted = computed(() => {
            const rank = { pending: 0, included: 1, excluded: 2 };
            return [...(selectedHandover.value?.inventory || [])]
                .sort((a, b) => (rank[a.status] ?? 9) - (rank[b.status] ?? 9));
        });
        const handoverInventoryPending = computed(() => handoverInventorySorted.value.filter(entry => entry.status === 'pending'));
        const handoverInventoryClosed = computed(() => handoverInventorySorted.value.filter(entry => entry.status !== 'pending'));

        const handoverItemsSorted = computed(() => [...(selectedHandover.value?.items || [])]
            .sort((a, b) => (handoverItemRank[a.status] ?? 9) - (handoverItemRank[b.status] ?? 9)));
        const handoverItemsActionable = computed(() => handoverItemsSorted.value.filter(item => item.status !== 'accepted'));
        const handoverItemsAccepted = computed(() => handoverItemsSorted.value.filter(item => item.status === 'accepted'));

        const handoverRisksSorted = computed(() => [...(selectedHandover.value?.risk_items || [])]
            .sort((a, b) => {
                const statusDelta = (handoverRiskStatusRank[a.status] ?? 9) - (handoverRiskStatusRank[b.status] ?? 9);
                if (statusDelta) return statusDelta;
                return (handoverRiskSeverityRank[a.severity] ?? 9) - (handoverRiskSeverityRank[b.severity] ?? 9);
            }));
        const handoverRisksActive = computed(() => handoverRisksSorted.value.filter(risk => risk.status !== 'closed'));
        const handoverRisksClosed = computed(() => handoverRisksSorted.value.filter(risk => risk.status === 'closed'));

        const handoverMyActions = computed(() => {
            const record = selectedHandover.value;
            if (!record || record.status === 'completed') return [];
            const items = record.items || [];
            const actions = [];
            if (canReviewHandover()) {
                const pending = handoverInventoryPending.value.length;
                if (pending) actions.push({ key: 'inventory', label: `${pending} 项盘点待审阅`, section: 'inventory' });
            }
            const rejected = items.filter(item => item.status === 'rejected' && canSubmitHandoverItem(item)).length;
            if (rejected) actions.push({ key: 'items-resubmit', label: `${rejected} 项已退回待补充`, section: 'items' });
            const toReview = items.filter(item => item.status === 'submitted' && canReviewHandoverItem(item)).length;
            if (toReview) actions.push({ key: 'items-review', label: `${toReview} 项待我验收`, section: 'items' });
            const toSubmit = items.filter(item => item.status === 'pending' && canSubmitHandoverItem(item)).length;
            if (toSubmit) actions.push({ key: 'items-submit', label: `${toSubmit} 项待我提交证据`, section: 'items' });
            const blocking = handoverRisksActive.value.filter(risk => risk.severity === 'blocking').length;
            if (blocking) actions.push({ key: 'risks-blocking', label: `${blocking} 项阻断风险未关闭`, section: 'risks' });
            const toClose = handoverRisksActive.value.filter(risk => risk.status === 'pending_close' && canReviewHandover()).length;
            if (toClose) actions.push({ key: 'risks-close', label: `${toClose} 项风险待确认关闭`, section: 'risks' });
            const toMitigate = handoverRisksActive.value.filter(
                risk => ['open', 'mitigating'].includes(risk.status) && canManageHandoverRisk(risk),
            ).length;
            if (toMitigate) actions.push({ key: 'risks-mitigate', label: `${toMitigate} 项风险待处理`, section: 'risks' });
            return actions;
        });

        function handoverSectionIsOpen(key) {
            return handoverSectionOpen.value[key] !== false;
        }

        function toggleHandoverSection(key) {
            handoverSectionOpen.value = { ...handoverSectionOpen.value, [key]: !handoverSectionIsOpen(key) };
        }

        function focusHandoverSection(key) {
            handoverSectionOpen.value = { ...handoverSectionOpen.value, [key]: true };
            nextTick(() => {
                document.getElementById(`handover-section-${key}`)?.scrollIntoView({ block: 'start', behavior: 'smooth' });
            });
        }

        function handoverClosedVisible(key) {
            return handoverShowClosed.value[key] === true;
        }

        function toggleHandoverClosed(key) {
            handoverShowClosed.value = { ...handoverShowClosed.value, [key]: !handoverClosedVisible(key) };
        }

        function handoverInventoryExcludeIsActive(entry) {
            return handoverInventoryExcludeActive.value[entry?.inventory_id] === true;
        }

        function beginHandoverInventoryExclude(entry) {
            handoverInventoryExcludeActive.value = { ...handoverInventoryExcludeActive.value, [entry.inventory_id]: true };
        }

        function cancelHandoverInventoryExclude(entry) {
            handoverInventoryExcludeActive.value = { ...handoverInventoryExcludeActive.value, [entry.inventory_id]: false };
        }

        function applyHandoverViewDefaults() {
            handoverSectionOpen.value = {
                inventory: handoverInventoryPending.value.length > 0,
                items: handoverItemsActionable.value.length > 0,
                risks: handoverRisksActive.value.length > 0 || canRegisterHandoverRisk(),
            };
            handoverShowClosed.value = { inventory: false, items: false, risks: false };
            handoverInventoryExcludeActive.value = {};
            handoverShowRiskForm.value = false;
        }

        // ── 知识库函数 ──

        async function loadCategories() {
            try {
                const res = await apiFetch(`${API_BASE}/categories`);
                const data = await res.json();
                if (data.success) categories.value = data.categories;
            } catch (e) { console.error('加载分类失败:', e); }
        }

        async function addCategory() {
            const name = newCatName.value.trim();
            if (!name) { alert('请输入分类名称'); return; }
            try {
                const form = new FormData();
                form.append('name', name);
                const res = await apiFetch(`${API_BASE}/categories`, { method: 'POST', body: form });
                const data = await res.json();
                if (data.success) {
                    categories.value = data.categories;
                    newCatName.value = '';
                } else {
                    alert(data.detail || '添加失败');
                }
            } catch (e) { alert('添加失败: ' + e.message); }
        }

        function startCatEdit(idx) {
            if (isBuiltInCategory(categories.value[idx])) return;
            editingCatIdx.value = idx;
            editingCatValue.value = categories.value[idx];
        }

        async function saveCatEdit(idx) {
            const oldName = categories.value[idx];
            const newName = editingCatValue.value.trim();
            if (!newName) { alert('分类名称不能为空'); return; }
            if (oldName === newName) { editingCatIdx.value = -1; return; }
            try {
                const form = new FormData();
                form.append('old_name', oldName);
                form.append('new_name', newName);
                const res = await apiFetch(`${API_BASE}/categories`, { method: 'PUT', body: form });
                const data = await res.json();
                if (data.success) {
                    categories.value = data.categories;
                    editingCatIdx.value = -1;
                } else {
                    alert(data.detail || '编辑失败');
                }
            } catch (e) { alert('编辑失败: ' + e.message); }
        }

        async function deleteCategory(name) {
            if (isBuiltInCategory(name)) {
                alert('内置分类用于自动归类和知识关联，不能直接删除');
                return;
            }
            if (!confirm(`确定删除分类「${categoryLabel(name)}」吗？`)) return;
            try {
                const res = await apiFetch(`${API_BASE}/categories/${encodeURIComponent(name)}`, { method: 'DELETE' });
                const data = await res.json();
                if (data.success) {
                    categories.value = data.categories;
                } else {
                    alert(data.detail || '删除失败');
                }
            } catch (e) { alert('删除失败: ' + e.message); }
        }

        const uploadUpdatableAssets = computed(() => knowledgeAssets.value.filter((asset) => {
            const status = knowledgeAssetStatus(asset);
            return knowledgeAssetId(asset) && !['revoked', 'deleted'].includes(status);
        }));

        const uploadTargetAsset = computed(() => uploadUpdatableAssets.value.find(
            asset => knowledgeAssetId(asset) === uploadTargetAssetId.value,
        ) || null);

        function resetKbUploadForm() {
            uploadMode.value = 'create';
            uploadTargetAssetId.value = '';
            uploadCategory.value = '';
            uploadFileData.value = null;
            uploadFilesData.value = [];
            uploadFileSelected.value = false;
            uploadFileName.value = '';
            graphImportHint.value = '';
            kbUploadMsg.value = '';
            uploaderName.value = currentUser.value?.display_name || '';
            uploadInputKey.value += 1;
        }

        async function openUploadModal(mode = 'create') {
            resetKbUploadForm();
            knowledgeTaskUploadContext.value = null;
            uploadMode.value = mode === 'update' ? 'update' : 'create';
            showUploadModal.value = true;
            if (!knowledgeAssets.value.length) await loadKnowledgeAssets();
        }

        function closeUploadModal() {
            if (kbUploading.value) return;
            showUploadModal.value = false;
            knowledgeTaskUploadContext.value = null;
            resetKbUploadForm();
        }

        function setUploadMode(mode) {
            if (knowledgeTaskUploadContext.value) {
                uploadMode.value = 'create';
                return;
            }
            uploadMode.value = mode === 'update' ? 'update' : 'create';
            uploadTargetAssetId.value = '';
            kbUploadMsg.value = '';
            uploadFileData.value = null;
            uploadFilesData.value = [];
            uploadFileSelected.value = false;
            uploadFileName.value = '';
            graphImportHint.value = '';
            uploadInputKey.value += 1;
        }

        function onKbFileChange(e) {
            const files = e.target.files;
            if (files && files.length > 0) {
                const selectedFiles = Array.from(files);
                const acceptedFiles = uploadMode.value === 'create' ? selectedFiles : selectedFiles.slice(0, 1);
                const file = acceptedFiles[0];
                uploadFilesData.value = acceptedFiles;
                uploadFileData.value = file;
                uploadFileSelected.value = true;
                uploadFileName.value = acceptedFiles.length > 1
                    ? `${acceptedFiles.length} 个文件：${acceptedFiles.slice(0, 3).map(item => item.name).join('、')}${acceptedFiles.length > 3 ? '…' : ''}`
                    : file.name;
                graphImportHint.value = '';
                if (acceptedFiles.length > 1) {
                    graphImportHint.value = '每个文件都会生成独立任务；单项失败、重试或取消不会阻断其他文件。';
                } else if (file.name.toLowerCase().endsWith('.json')) {
                    file.text().then((content) => {
                        try {
                            const payload = JSON.parse(content);
                            const nodes = payload.nodes || payload.vertices || payload.items || [];
                            const edges = payload.edges || payload.links || payload.relations || [];
                            const countItems = (value) => Array.isArray(value)
                                ? value.length
                                : (value && typeof value === 'object'
                                    ? Object.values(value).reduce((total, item) => total + (Array.isArray(item) ? item.length : 0), 0)
                                    : 0);
                            const nodeCount = countItems(nodes);
                            const edgeCount = countItems(edges);
                            graphImportHint.value = nodeCount
                                ? `识别为图谱 JSON：${nodeCount} 个节点 / ${edgeCount} 条关系，上传后将直接更新知识图谱。`
                                : 'JSON 格式有效，但未识别到 nodes / edges；将作为普通知识文档入库。';
                        } catch (error) {
                            graphImportHint.value = 'JSON 格式无效，上传前请检查文件内容。';
                        }
                    });
                }
            }
        }

        async function submitKbUpload() {
            if (!uploadFileSelected.value) { alert('请选择文件'); return; }
            if (!uploaderName.value.trim()) { alert('请输入上传者姓名'); return; }
            if (uploadMode.value === 'update' && !uploadTargetAsset.value) {
                kbUploadMsg.value = '失败：请选择要更新的现有知识资产';
                return;
            }
            if (uploadMode.value === 'update' && !knowledgeAssetSourceRecord(uploadTargetAsset.value)?.source_id
                && !uploadTargetAsset.value?.primary_source_id) {
                kbUploadMsg.value = '失败：目标资产缺少稳定来源标识，暂时无法追加新版本';
                return;
            }
            kbUploading.value = true;
            kbUploadMsg.value = '';
            activeIngestionStep.value = 0;
            try {
                const form = new FormData();
                form.append('uploader', uploaderName.value.trim());
                if (uploadCategory.value) form.append('category', uploadCategory.value);
                const isBatch = uploadMode.value === 'create' && uploadFilesData.value.length > 1;
                if (isBatch) {
                    uploadFilesData.value.forEach(file => form.append('files', file));
                } else {
                    form.append('file', uploadFileData.value);
                }
                if (!isBatch && uploadMode.value === 'update') {
                    form.append('asset_id', knowledgeAssetId(uploadTargetAsset.value));
                    form.append('source_id', knowledgeAssetSourceRecord(uploadTargetAsset.value)?.source_id
                        || uploadTargetAsset.value.primary_source_id);
                }
                const endpoint = isBatch ? 'documents/upload-batches' : 'documents/uploads';
                const res = await apiFetch(`${API_BASE}/${endpoint}`, { method: 'POST', body: form });
                const data = await res.json().catch(() => ({}));
                if (res.ok && data.success) {
                    const acceptedJob = data.job;
                    const acceptedFilename = data.filename || uploadFileName.value;
                    const acceptedJobs = data.jobs || (acceptedJob ? [acceptedJob] : []);
                    const taskUploadContext = knowledgeTaskUploadContext.value;
                    const acceptedFilenames = Array.isArray(data.items)
                        ? data.items.filter(item => item.status === 'accepted').map(item => item.filename).filter(Boolean)
                        : (isBatch ? uploadFilesData.value.map(file => file.name) : [acceptedFilename]);
                    if (taskUploadContext) {
                        recordKnowledgeTaskUpload(taskUploadContext, acceptedFilenames);
                    }
                    showUploadModal.value = false;
                    knowledgeTaskUploadContext.value = null;
                    resetKbUploadForm();
                    processingJobStatusFilter.value = 'active';
                    processingJobTypeFilter.value = 'document_ingestion';
                    processingJobKeyword.value = '';
                    processingJobMessage.value = data.message || `${acceptedFilename} 已受理，正在后台入库`;
                    if (!taskUploadContext) tab.value = 'jobs';
                    await loadProcessingJobs({ keepSelection: !taskUploadContext });
                    if (isBatch && acceptedJobs.length) {
                        trackDocumentUploadBatchJobs(acceptedJobs, data.submitted_count || acceptedJobs.length);
                    } else if (acceptedJob?.job_id) {
                        if (!taskUploadContext) await openProcessingJob(acceptedJob, { silent: true });
                        trackDocumentUploadJob(acceptedJob.job_id, acceptedFilename);
                    }
                } else {
                    kbUploadMsg.value = '失败：' + (data.detail || '上传失败');
                }
            } catch (e) {
                kbUploadMsg.value = '失败：上传失败: ' + e.message;
            } finally {
                kbUploading.value = false;
            }
        }

        async function trackDocumentUploadBatchJobs(jobs, submittedCount) {
            const pending = new Set(jobs.map(job => job?.job_id).filter(Boolean));
            const terminal = { succeeded: 0, failed: 0, cancelled: 0 };
            let consecutiveErrors = 0;
            for (let attempt = 0; attempt < 240 && pending.size; attempt += 1) {
                await new Promise(resolve => setTimeout(resolve, 1500));
                try {
                    const states = await Promise.all([...pending].map(async (jobId) => {
                        const res = await apiFetch(`${API_BASE}/processing/jobs/${encodeURIComponent(jobId)}`);
                        const data = await res.json().catch(() => ({}));
                        if (!res.ok || !data.success) throw new Error(data.detail || '任务状态读取失败');
                        return data.job;
                    }));
                    consecutiveErrors = 0;
                    states.forEach((job) => {
                        if (selectedProcessingJob.value?.job_id === job.job_id) selectedProcessingJob.value = job;
                        if (['succeeded', 'failed', 'cancelled'].includes(job.status) && pending.delete(job.job_id)) {
                            terminal[job.status] += 1;
                        }
                    });
                    processingJobMessage.value = pending.size
                        ? `批量入库进度：已结束 ${jobs.length - pending.size} / ${jobs.length} 项`
                        : `批量入库完成：成功 ${terminal.succeeded} 项，失败 ${terminal.failed} 项，取消 ${terminal.cancelled} 项`;
                    await loadProcessingJobs({ silent: true });
                } catch (error) {
                    consecutiveErrors += 1;
                    if (consecutiveErrors >= 3) {
                        processingJobMessage.value = '批量任务状态暂时无法读取，可稍后刷新任务中心';
                        return;
                    }
                }
            }
            if (!pending.size) {
                processingJobStatusFilter.value = '';
                await Promise.all([loadGroupedDocs(), loadKnowledgeAssets(), loadStats(), loadGraph()]);
                await loadProcessingJobs({ silent: true });
            } else {
                processingJobMessage.value = `${submittedCount} 个文件仍在处理中，可稍后在任务中心继续查看`;
            }
        }

        async function trackDocumentUploadJob(jobId, filename) {
            let consecutiveErrors = 0;
            for (let attempt = 0; attempt < 300; attempt += 1) {
                await new Promise(resolve => setTimeout(resolve, 1000));
                try {
                    const res = await apiFetch(`${API_BASE}/processing/jobs/${encodeURIComponent(jobId)}`);
                    const data = await res.json().catch(() => ({}));
                    if (!res.ok || !data.success) throw new Error(data.detail || '任务状态读取失败');
                    consecutiveErrors = 0;
                    const job = data.job;
                    if (selectedProcessingJob.value?.job_id === jobId) selectedProcessingJob.value = job;
                    if (!['succeeded', 'failed', 'cancelled'].includes(job.status)) continue;
                    processingJobStatusFilter.value = '';
                    if (job.status === 'succeeded') {
                        const result = job.result || {};
                        const syncWarnings = ingestionSyncWarnings(result);
                        const extractionWarning = result.extraction_warning || '';
                        lastIngestion.value = {
                            assetId: knowledgeAssetId(result.asset), filename: result.filename || filename,
                            chunks: result.chunks || 0, categories: result.categories || [],
                            graphImported: !!result.graph_imported, graphNodes: result.graph_nodes || 0,
                            graphEdges: result.graph_edges || 0, graphSynced: result.graph_synced !== false,
                            contextSynced: result.context_synced !== false,
                            readinessSynced: result.readiness_synced !== false,
                            projectionStatus: result.projection_status || '',
                            syncWarning: syncWarnings.length > 0, syncWarnings,
                        };
                        processingJobMessage.value = extractionWarning
                            ? `${filename} 已入库，OCR 提示：${extractionWarning}`
                            : syncWarnings.length
                            ? `${filename} 已入库，派生同步待修复：${syncWarnings.join('、')}`
                            : `${filename} 已完成知识入库`;
                        await Promise.all([loadGroupedDocs(), loadKnowledgeAssets(), loadStats(), loadGraph()]);
                    } else {
                        processingJobMessage.value = job.status === 'cancelled'
                            ? `${filename} 的入库任务已取消`
                            : `${filename} 入库失败：${job.last_error || '请查看尝试记录后重试'}`;
                    }
                    await loadProcessingJobs({ silent: true });
                    return;
                } catch (error) {
                    consecutiveErrors += 1;
                    if (consecutiveErrors >= 5) {
                        processingJobMessage.value = `${filename} 的任务状态暂时无法读取，可稍后刷新任务中心`;
                        return;
                    }
                }
            }
            processingJobMessage.value = `${filename} 仍在处理中，可稍后在任务中心继续查看`;
        }

        async function loadGroupedDocs() {
            kbLoading.value = true;
            kbLoadError.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/documents/by-category`);
                const data = await res.json();
                if (!res.ok || !data.success) throw new Error(data.detail || '知识库加载失败');
                groupedDocs.value = data.grouped || {};
                if (data.categories) categories.value = data.categories;
            } catch (e) {
                groupedDocs.value = {};
                kbLoadError.value = '知识库暂时无法读取：' + e.message;
            } finally {
                kbLoading.value = false;
            }
        }

        async function loadKnowledgeAssets() {
            knowledgeAssetsLoading.value = true;
            knowledgeAssetsError.value = '';
            knowledgeAssetsAccessDenied.value = false;
            try {
                const query = can('knowledge.manage')
                    ? '?include_inactive=true&include_deleted=false'
                    : '';
                const res = await apiFetch(`${API_BASE}/knowledge/assets${query}`);
                const data = await res.json().catch(() => ({}));
                if (res.status === 403) {
                    knowledgeAssetsAccessDenied.value = true;
                    knowledgeAssets.value = [];
                    knowledgeAssetSummary.value = {};
                    return;
                }
                if (!res.ok) throw new Error(data.detail || '知识资产加载失败');
                knowledgeAssets.value = Array.isArray(data.assets) ? data.assets : [];
                knowledgeAssetSummary.value = data.summary || {};
            } catch (error) {
                knowledgeAssets.value = [];
                knowledgeAssetSummary.value = {};
                knowledgeAssetsError.value = `知识资产暂时无法读取：${error.message || '未知错误'}`;
            } finally {
                knowledgeAssetsLoading.value = false;
            }
        }

        function syncKnowledgeAssetDraft(asset) {
            knowledgeAssetDraft.value = {
                title: knowledgeAssetTitle(asset),
                owner_user_id: asset?.owner_user_id || asset?.owner?.user_id || '',
                review_due_at: knowledgeReviewDate(asset) ? String(knowledgeReviewDate(asset)).slice(0, 10) : '',
                visibility: knowledgeVisibility(asset),
            };
        }

        function clearKnowledgeAssetDetail(status, assetId = '') {
            selectedKnowledgeAsset.value = { asset_id: assetId };
            knowledgeAssetVersions.value = [];
            knowledgeAssetDetailUnavailable.value = status;
            knowledgeAssetDetailAccessDenied.value = status === 'denied';
            knowledgeAssetDetailError.value = status === 'not_found'
                ? '该知识资产已不存在，或已被移出当前项目。'
                : '';
            knowledgeAssetMessage.value = '';
            knowledgeRevokeReason.value = '';
            syncKnowledgeAssetDraft({});
        }

        function handleKnowledgeDetailResponse(res, assetId) {
            if (res.status === 403) {
                clearKnowledgeAssetDetail('denied', assetId);
                return true;
            }
            if (res.status === 404) {
                clearKnowledgeAssetDetail('not_found', assetId);
                return true;
            }
            return false;
        }

        async function openKnowledgeAsset(asset) {
            selectedKnowledgeAsset.value = asset;
            knowledgeAssetVersions.value = [];
            knowledgeAssetDetailLoading.value = false;
            knowledgeAssetDetailError.value = '';
            knowledgeAssetDetailAccessDenied.value = false;
            knowledgeAssetDetailUnavailable.value = '';
            knowledgeAssetMessage.value = '';
            knowledgeRevokeReason.value = '';
            syncKnowledgeAssetDraft(asset);
            const assetId = knowledgeAssetId(asset);
            if (!assetId) {
                knowledgeAssetDetailError.value = '该记录缺少稳定资产 ID，暂时无法读取版本历史。';
                return;
            }
            knowledgeAssetDetailLoading.value = true;
            try {
                const detailRes = await apiFetch(`${API_BASE}/knowledge/assets/${encodeURIComponent(assetId)}`);
                const detailData = await detailRes.json().catch(() => ({}));
                if (handleKnowledgeDetailResponse(detailRes, assetId)) return;
                if (!detailRes.ok) throw new Error(detailData.detail || '资产详情加载失败');

                const versionRes = await apiFetch(`${API_BASE}/knowledge/assets/${encodeURIComponent(assetId)}/versions`);
                const versionData = await versionRes.json().catch(() => ({}));
                if (handleKnowledgeDetailResponse(versionRes, assetId)) return;
                if (!versionRes.ok) throw new Error(versionData.detail || '版本历史加载失败');

                selectedKnowledgeAsset.value = detailData.asset || detailData;
                knowledgeAssetVersions.value = Array.isArray(versionData.versions) ? versionData.versions : [];
                const currentVersion = knowledgeCurrentVersion(selectedKnowledgeAsset.value);
                if (currentVersion && !selectedKnowledgeAsset.value.current_version) {
                    selectedKnowledgeAsset.value = { ...selectedKnowledgeAsset.value, current_version: currentVersion };
                }
                syncKnowledgeAssetDraft(selectedKnowledgeAsset.value);
            } catch (error) {
                knowledgeAssetDetailError.value = error.message || '版本历史加载失败';
            } finally {
                knowledgeAssetDetailLoading.value = false;
            }
        }

        function closeKnowledgeAsset() {
            selectedKnowledgeAsset.value = null;
            knowledgeAssetVersions.value = [];
            knowledgeAssetDetailError.value = '';
            knowledgeAssetDetailAccessDenied.value = false;
            knowledgeAssetDetailUnavailable.value = '';
            knowledgeAssetMessage.value = '';
            knowledgeRevokeReason.value = '';
            knowledgeRevokeConfirmOpen.value = false;
        }

        function previewKnowledgeAssetSource(asset) {
            const source = knowledgeVersionDocuments(knowledgeCurrentVersion(asset))[0] || knowledgeAssetSourceRecord(asset);
            if (!source?.stored_file && !source?.source_file && !source?.name) {
                knowledgeAssetMessage.value = '当前版本没有可预览的来源文件';
                return;
            }
            previewDoc(source);
        }

        async function saveKnowledgeAsset() {
            if (!can('knowledge.manage') || !selectedKnowledgeAsset.value) return;
            const assetId = knowledgeAssetId(selectedKnowledgeAsset.value);
            if (!assetId) return;
            if (knowledgeAssetDraft.value.visibility === 'private' && !knowledgeAssetDraft.value.owner_user_id) {
                knowledgeAssetMessage.value = '设置为仅责任人可见前，必须先选择责任人';
                return;
            }
            knowledgeAssetSaving.value = true;
            knowledgeAssetMessage.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/knowledge/assets/${encodeURIComponent(assetId)}`, {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        title: knowledgeAssetDraft.value.title,
                        owner_user_id: knowledgeAssetDraft.value.owner_user_id,
                        review_due_at: knowledgeAssetDraft.value.review_due_at,
                        visibility: knowledgeAssetDraft.value.visibility,
                    }),
                });
                const data = await res.json().catch(() => ({}));
                if (handleKnowledgeDetailResponse(res, assetId)) return;
                if (!res.ok) throw new Error(data.detail || '资产信息保存失败');
                selectedKnowledgeAsset.value = { ...selectedKnowledgeAsset.value, ...(data.asset || data) };
                syncKnowledgeAssetDraft(selectedKnowledgeAsset.value);
                knowledgeAssetMessage.value = '资产信息已保存';
                await loadKnowledgeAssets();
            } catch (error) {
                knowledgeAssetMessage.value = `保存失败：${error.message || '未知错误'}`;
            } finally {
                knowledgeAssetSaving.value = false;
            }
        }

        function knowledgeAssetNeedsRepair(asset) {
            return knowledgeProjectionItems(knowledgeCurrentVersion(asset))
                .some(item => ['repair_required', 'failed'].includes(item.status));
        }

        async function repairKnowledgeAssetProjections() {
            if (!can('knowledge.manage') || !selectedKnowledgeAsset.value) return;
            const assetId = knowledgeAssetId(selectedKnowledgeAsset.value);
            if (!assetId) return;
            knowledgeAssetSaving.value = true;
            knowledgeAssetMessage.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/knowledge/assets/${encodeURIComponent(assetId)}/repair-projections`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ force: false }),
                });
                const data = await res.json().catch(() => ({}));
                if (handleKnowledgeDetailResponse(res, assetId)) return;
                if (!res.ok) throw new Error(data.detail || '投影修复失败');
                knowledgeAssetMessage.value = `${data.message || '投影修复已进入处理队列'}${data.job?.job_id ? ` · ${data.job.job_id}` : ''}`;
                await loadProcessingJobs({ keepSelection: false, silent: true });
            } catch (error) {
                knowledgeAssetMessage.value = `修复失败：${error.message || '未知错误'}`;
            } finally {
                knowledgeAssetSaving.value = false;
            }
        }

        const knowledgeTransitionContract = {
            submit_review: { backendAction: 'submit_review', target: 'pending_review' },
            return_to_draft: { backendAction: 'return_to_draft', target: 'draft' },
            publish: { backendAction: 'publish', target: 'active' },
            mark_review_due: { backendAction: 'mark_review_due', target: 'review_due' },
            confirm_valid: { backendAction: 'confirm_valid', target: 'active' },
            expire: { backendAction: 'expire', target: 'expired' },
            revoke: { backendAction: 'revoke', target: 'revoked' },
        };

        function knowledgeLatestReadyVersion() {
            return [...knowledgeAssetVersions.value]
                .filter(version => String(version?.status || '').toLowerCase() === 'ready')
                .sort((left, right) => Number(right?.version_no || 0) - Number(left?.version_no || 0))[0] || null;
        }

        function knowledgeAvailableTransitions(asset) {
            const status = knowledgeAssetStatus(asset);
            const map = {
                draft: [
                    { action: 'submit_review', label: '提交审核', tone: 'primary' },
                    { action: 'revoke', label: '撤销资产', tone: 'danger' },
                ],
                pending_review: [
                    { action: 'publish', label: '发布生效', tone: 'primary', disabled: !knowledgeLatestReadyVersion(), disabledReason: '没有可发布的 ready 版本' },
                    { action: 'return_to_draft', label: '退回草稿', tone: 'secondary' },
                    { action: 'revoke', label: '撤销资产', tone: 'danger' },
                ],
                active: [
                    { action: 'mark_review_due', label: '标记待复审', tone: 'secondary' },
                    { action: 'expire', label: '设为失效', tone: 'secondary' },
                    { action: 'revoke', label: '撤销资产', tone: 'danger' },
                ],
                review_due: [
                    { action: 'confirm_valid', label: '确认有效', tone: 'primary' },
                    { action: 'expire', label: '设为失效', tone: 'secondary' },
                    { action: 'revoke', label: '撤销资产', tone: 'danger' },
                ],
                expired: [
                    { action: 'revoke', label: '撤销资产', tone: 'danger' },
                ],
                revoked: [],
            };
            return map[status] || [];
        }

        async function transitionKnowledgeAsset(action) {
            if (!can('knowledge.publish') || !selectedKnowledgeAsset.value) return;
            const assetId = knowledgeAssetId(selectedKnowledgeAsset.value);
            if (!assetId) return;
            const contract = knowledgeTransitionContract[action];
            if (!contract || contract.unsupported) {
                knowledgeAssetMessage.value = '当前服务尚未开放该生命周期操作';
                return;
            }
            if (action === 'revoke' && !knowledgeRevokeReason.value.trim()) {
                knowledgeAssetMessage.value = '撤销资产前必须填写撤销原因';
                return;
            }
            if (action === 'revoke' && !knowledgeRevokeConfirmOpen.value) {
                knowledgeRevokeConfirmOpen.value = true;
                return;
            }
            const readyVersion = action === 'publish' ? knowledgeLatestReadyVersion() : null;
            if (action === 'publish' && !readyVersion?.version_id) {
                knowledgeAssetMessage.value = '没有可发布的 ready 版本，请先完成版本处理';
                return;
            }
            knowledgeAssetSaving.value = true;
            knowledgeAssetMessage.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/knowledge/assets/${encodeURIComponent(assetId)}/transition`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        action: contract.backendAction,
                        ...(readyVersion?.version_id ? { version_id: readyVersion.version_id } : {}),
                        ...(action === 'revoke' ? { reason: knowledgeRevokeReason.value.trim() } : {}),
                        expected_revision: selectedKnowledgeAsset.value.lifecycle_revision,
                    }),
                });
                const data = await res.json().catch(() => ({}));
                if (handleKnowledgeDetailResponse(res, assetId)) {
                    knowledgeRevokeConfirmOpen.value = false;
                    return;
                }
                if (!res.ok) throw new Error(data.detail || '状态变更失败');
                selectedKnowledgeAsset.value = { ...selectedKnowledgeAsset.value, ...(data.asset || data) };
                syncKnowledgeAssetDraft(selectedKnowledgeAsset.value);
                knowledgeRevokeReason.value = '';
                knowledgeRevokeConfirmOpen.value = false;
                let successMessage = `状态已更新为“${knowledgeStatusLabel(selectedKnowledgeAsset.value)}”`;
                if (action === 'submit_review') {
                    successMessage = data.task_sync?.task_id
                        ? `${successMessage}；发布审核任务已创建`
                        : `${successMessage}；发布任务创建失败，请在知识任务中心补建`;
                }
                await loadKnowledgeAssets();
                await openKnowledgeAsset(selectedKnowledgeAsset.value);
                if (!knowledgeAssetDetailUnavailable.value) knowledgeAssetMessage.value = successMessage;
            } catch (error) {
                knowledgeRevokeConfirmOpen.value = false;
                knowledgeAssetMessage.value = `操作失败：${error.message || '未知错误'}`;
            } finally {
                knowledgeAssetSaving.value = false;
            }
        }

        function knowledgeVersionName(version, index) {
            const value = version?.version_no ?? version?.version_number ?? version?.number ?? version?.label;
            return value !== undefined && value !== null && value !== '' ? `v${String(value).replace(/^v/i, '')}` : `版本 ${knowledgeAssetVersions.value.length - index}`;
        }

        function isKnowledgeCurrentVersion(version) {
            return String(version?.version_id || '') === String(selectedKnowledgeAsset.value?.current_version_id || '')
                || Boolean(version?.current || version?.is_current);
        }

        function knowledgeVersionStatus(version) {
            const status = String(version?.processing_status || version?.status || 'completed').toLowerCase();
            return {
                completed: '处理完成', active: '当前生效', ready: '待发布', preparing: '准备中', processing: '处理中',
                pending: '等待处理', failed: '处理失败', superseded: '历史版本', revoked: '已撤销',
            }[status] || status;
        }

        async function previewKnowledgeVersion(version) {
            const assetId = knowledgeAssetId(selectedKnowledgeAsset.value);
            const versionId = String(version?.version_id || '');
            if (!assetId || !versionId || knowledgeVersionPreviewingId.value) return;
            knowledgeVersionPreviewingId.value = versionId;
            previewFilename.value = `${knowledgeAssetTitle(selectedKnowledgeAsset.value)} · ${knowledgeVersionName(version, 0)}`;
            previewData.value = '正在读取此版本内容…';
            try {
                const res = await apiFetch(`${API_BASE}/knowledge/assets/${encodeURIComponent(assetId)}/versions/${encodeURIComponent(versionId)}/preview`);
                const data = await res.json().catch(() => ({}));
                if (res.status === 403) {
                    clearKnowledgeAssetDetail('denied', assetId);
                    previewData.value = null;
                    return;
                }
                if (!res.ok || data.success === false) throw new Error(data.detail || '该版本暂时无法预览');
                const result = data.preview && typeof data.preview === 'object' ? data.preview : data;
                previewFilename.value = result.filename || result.title || previewFilename.value;
                previewData.value = result.content || result.text || result.markdown || '该版本没有可展示的文本内容';
                if (result.truncated) previewData.value += '\n\n[内容较长，仅展示部分内容]';
            } catch (error) {
                previewData.value = `无法预览此版本：${error.message || '未知错误'}`;
            } finally {
                knowledgeVersionPreviewingId.value = '';
            }
        }

        async function previewDoc(doc) {
            const displayName = doc.source_file || doc.name || doc.stored_file;
            const filename = doc.stored_file || doc.name || doc.source_file;
            previewFilename.value = displayName;
            previewData.value = '加载中…';
            try {
                const res = await apiFetch(`${API_BASE}/documents/preview/${encodeURIComponent(filename)}`);
                const data = await res.json();
                if (!res.ok || !data.success) throw new Error(data.detail || '无法预览该文件');
                previewData.value = data.content + (data.truncated ? '\n\n[内容较长，仅展示前 30000 个字符]' : '');
            } catch (e) {
                previewData.value = '预览失败: ' + e.message;
            }
        }

        // ── 对话函数 ──

        function renderMarkdown(text) {
            if (!text) return '';
            // Step 1: 保护代码块
            const codeBlocks = [];
            text = text.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
                const idx = codeBlocks.length;
                codeBlocks.push(`<pre class="code-block" data-language="${lang || 'text'}"><code class="lang-${lang || 'text'}">${code
                    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                }</code></pre>`);
                return `%%CODEBLOCK${idx}%%`;
            });
            // 保护行内代码
            const inlineCodes = [];
            text = text.replace(/`([^`]+)`/g, (_, code) => {
                const idx = inlineCodes.length;
                inlineCodes.push(`<code>${code
                    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                }</code>`);
                return `%%INLINECODE${idx}%%`;
            });
            // Step 2: 转义 HTML
            text = text
                .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            // Step 3: 处理 Markdown 块级语法
            // 标题
            text = text.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
            text = text.replace(/^### (.+)$/gm, '<h3>$1</h3>');
            text = text.replace(/^## (.+)$/gm, '<h2>$1</h2>');
            text = text.replace(/^# (.+)$/gm, '<h1>$1</h1>');
            // 分割线
            text = text.replace(/^(?:---|\*\*\*|___)\s*$/gm, '<hr>');
            // 列表：先转义标记，后续用换行识别
            // 有序列表
            text = text.replace(/^\s*\d+\.\s+(.+)$/gm, '<oli>$1</oli>');
            text = text.replace(/(<oli>.*<\/oli>(\s*<oli>.*<\/oli>)*)/g, '<ol>$1</ol>');
            text = text.replace(/<\/?oli>/g, (m) => m.replace('oli', 'li'));
            // 无序列表
            text = text.replace(/^[\s]*[-*+] (.+)$/gm, '<li>$1</li>');
            text = text.replace(/(<li>.*<\/li>(\s*<li>.*<\/li>)*)/g, '<ul>$1</ul>');
            // Step 4: 处理行内语法
            // 加粗
            text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
            // 斜体
            text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');
            // 链接
            text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
            // Step 5: 段落和换行
            text = text.replace(/\n\n+/g, '</p><p>');
            text = text.replace(/\n/g, '<br>');
            text = '<p>' + text + '</p>';
            // Step 6: 恢复保护的代码块和行内代码
            text = text.replace(/%%CODEBLOCK(\d+)%%/g, (_, idx) => codeBlocks[+idx] || '');
            text = text.replace(/%%INLINECODE(\d+)%%/g, (_, idx) => inlineCodes[+idx] || '');
            // 清理空段落
            text = text.replace(/<p>\s*<\/p>/g, '');
            return text;
        }

        function scrollToBottom() {
            nextTick(() => {
                if (messagesRef.value) {
                    messagesRef.value.scrollTop = messagesRef.value.scrollHeight;
                }
            });
        }

        function formatProcessDuration(seconds = 0) {
            const total = Math.max(1, Math.floor(Number(seconds) || 0));
            const minutes = Math.floor(total / 60);
            const remain = total % 60;
            return minutes ? `${minutes} 分 ${remain} 秒` : `${remain} 秒`;
        }

        function processStatusLabel(status) {
            return { active: '进行中', done: '已完成', waiting: '等待中', error: '未完成' }[status] || '处理中';
        }

        function compactProcessQuestion(question) {
            const normalized = String(question || '').replace(/\s+/g, ' ').trim();
            return normalized.length > 42 ? `${normalized.slice(0, 42)}...` : normalized;
        }

        function createAnalysisProcess(question, ragEnabled, smartEnabled = false) {
            const focus = compactProcessQuestion(question);
            const entries = smartEnabled
                ? [
                    { id: 'agent-routing', title: '分析问题并选择检索路径', detail: `正在识别“${focus}”并选择适合的项目知识能力进行核验。`, status: 'active', evidence: [] },
                    { id: 'answer', title: '组织回答', detail: '等待工具核验完成后生成回答。', status: 'waiting', evidence: [] },
                ]
                : ragEnabled
                ? [
                    { id: 'analysis', title: '理解问题', detail: `正在识别“${focus}”的关注点和回答范围。`, status: 'active', evidence: [] },
                    { id: 'retrieval', title: '检索项目资料', detail: '等待从项目文档、交接资料和知识图谱中提取相关内容。', status: 'waiting', evidence: [] },
                    { id: 'evidence', title: '核对证据', detail: '等待确认能够支持结论的文档片段。', status: 'waiting', evidence: [] },
                    { id: 'citation', title: '关联引用', detail: '等待生成可追溯的资料引用。', status: 'waiting', evidence: [] },
                    { id: 'answer', title: '组织回答', detail: '等待基于已核对资料生成结构化结论。', status: 'waiting', evidence: [] },
                ]
                : [
                    { id: 'analysis', title: '理解问题', detail: `正在分析“${focus}”的意图与回答边界。`, status: 'active', evidence: [] },
                    { id: 'answer', title: '组织回答', detail: '等待整理结论并生成清晰回复。', status: 'waiting', evidence: [] },
                ];
            return {
                entries,
                currentDetail: entries[0].detail,
                elapsedSeconds: 0,
                useRag: ragEnabled,
                smartMode: smartEnabled,
                running: true,
                complete: false,
                failed: false,
                expanded: true,
                summary: '',
            };
        }

        function updateAnalysisProcess(message, trace) {
            if (!message?.process || !trace?.id) return;
            const process = message.process;
            const evidence = Array.isArray(trace.evidence) ? trace.evidence.filter(Boolean) : [];
            let entry = process.entries.find(item => item.id === trace.id);
            if (!entry) {
                entry = { id: trace.id, title: trace.title || '分析步骤', detail: '', status: 'waiting', evidence: [] };
                const answerIndex = process.entries.findIndex(item => item.id === 'answer');
                if (trace.id !== 'answer' && answerIndex >= 0) process.entries.splice(answerIndex, 0, entry);
                else process.entries.push(entry);
            }
            entry.title = trace.title || entry.title;
            entry.detail = trace.detail || entry.detail;
            entry.status = trace.status || entry.status;
            if (evidence.length) entry.evidence = evidence;
            if (entry.status === 'active') process.currentDetail = entry.detail;

            const traceStepIndex = {
                'agent-routing': 0,
                'get_project_overview': 0,
                'search_project_knowledge': 0,
                'trace_knowledge_graph': 1,
                'get_onboarding_guide': 1,
                'get_handover_status': 1,
                'agent-fallback': 0,
                retrieval: 0,
                evidence: 1,
                answer: 2,
                citation: 3,
            };
            if (traceStepIndex[trace.id] !== undefined) {
                activeRetrievalStep.value = Math.max(activeRetrievalStep.value, traceStepIndex[trace.id]);
            }
        }

        function toggleThinking(msg) {
            if (!msg.process) return;
            msg.process.expanded = !msg.process.expanded;
        }

        function finishThinking(message, sourceCount = 0, failed = false) {
            if (!message.process) return;
            const process = message.process;
            process.running = false;
            process.complete = !failed;
            process.failed = failed;
            process.entries.forEach((entry) => {
                if (failed && entry.status === 'active') entry.status = 'error';
                else if (!failed && entry.status !== 'error') entry.status = 'done';
            });
            const answerEntry = process.entries.find(entry => entry.id === 'answer');
            if (answerEntry) {
                answerEntry.detail = failed
                    ? '未收到有效模型回答，请检查模型连接或稍后重试。'
                    : '已完成回答组织，正文与引用已返回。';
            }
            process.expanded = failed;
            process.summary = failed
                ? '处理未完成，可检查服务配置后重试。'
                : process.smartMode
                    ? `智能模式已完成项目知识核验，共关联 ${sourceCount} 条资料。`
                : process.useRag
                    ? `已核对 ${sourceCount} 条项目资料，结论与引用已关联。`
                    : '已完成问题分析与回答组织。';
        }

        async function submitAnswerFeedback(message, helpful) {
            if (!message?.answerId || message.feedbackSaving) return;
            if (!helpful && !message.feedbackReason) {
                message.feedbackOpen = true;
                return;
            }
            message.feedbackSaving = true;
            message.feedbackMessage = '';
            try {
                const res = await apiFetch(`${API_BASE}/rag/feedback`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        answer_id: message.answerId,
                        question: message.question || '',
                        helpful: Boolean(helpful),
                        reason: helpful ? '' : message.feedbackReason,
                        note: helpful ? '' : (message.feedbackNote || ''),
                        citations: message.sources || [],
                    }),
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '反馈保存失败');
                message.feedback = helpful ? 'helpful' : 'unhelpful';
                message.feedbackOpen = false;
                message.feedbackMessage = data.message || '感谢反馈';
                if (data.knowledge_task?.task_id) message.feedbackTaskId = data.knowledge_task.task_id;
            } catch (error) {
                message.feedbackMessage = error.message;
            } finally {
                message.feedbackSaving = false;
            }
        }

        async function loadRagEvaluationSets() {
            ragEvaluationLoading.value = true;
            ragEvaluationMessage.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/rag/evaluation-sets`);
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '评测集加载失败');
                ragEvaluationSets.value = data.evaluation_sets || [];
                if (selectedRagEvaluationSet.value?.evaluation_set_id) {
                    await selectRagEvaluationSet(selectedRagEvaluationSet.value.evaluation_set_id);
                }
            } catch (error) {
                ragEvaluationMessage.value = error.message;
            } finally {
                ragEvaluationLoading.value = false;
            }
        }

        async function openRagEvaluation() {
            ragEvaluationOpen.value = true;
            await loadRagEvaluationSets();
            if (!selectedRagEvaluationSet.value && ragEvaluationSets.value.length) {
                await selectRagEvaluationSet(ragEvaluationSets.value[0].evaluation_set_id);
            }
        }

        async function selectRagEvaluationSet(evaluationSetId) {
            try {
                const res = await apiFetch(`${API_BASE}/rag/evaluation-sets/${encodeURIComponent(evaluationSetId)}`);
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '评测集详情加载失败');
                selectedRagEvaluationSet.value = data.evaluation_set;
            } catch (error) {
                ragEvaluationMessage.value = error.message;
            }
        }

        async function createRagEvaluationSet() {
            if (!ragEvaluationSetDraft.value.name.trim() || ragEvaluationLoading.value) return;
            ragEvaluationLoading.value = true;
            try {
                const res = await apiFetch(`${API_BASE}/rag/evaluation-sets`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ...ragEvaluationSetDraft.value, min_case_target: 30 }),
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '评测集创建失败');
                ragEvaluationSetDraft.value = { name: '', version: 1, description: '' };
                await loadRagEvaluationSets();
                await selectRagEvaluationSet(data.evaluation_set.evaluation_set_id);
                ragEvaluationMessage.value = '评测集已创建，请持续补充到至少 30 题。';
            } catch (error) {
                ragEvaluationMessage.value = error.message;
            } finally {
                ragEvaluationLoading.value = false;
            }
        }

        async function addRagEvaluationCase() {
            const evaluationSetId = selectedRagEvaluationSet.value?.evaluation_set_id;
            const draft = ragEvaluationCaseDraft.value;
            if (!evaluationSetId || !draft.question.trim() || ragEvaluationLoading.value) return;
            ragEvaluationLoading.value = true;
            try {
                const splitLines = value => String(value || '').split('\n').map(item => item.trim()).filter(Boolean);
                const res = await apiFetch(`${API_BASE}/rag/evaluation-sets/${encodeURIComponent(evaluationSetId)}/cases`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        question: draft.question,
                        scenario: draft.expectNoAnswer ? 'no_answer' : draft.scenario,
                        expect_no_answer: draft.expectNoAnswer,
                        expected_points: splitLines(draft.expectedPoints),
                        allowed_source_files: splitLines(draft.allowedSources),
                    }),
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '评测问题保存失败');
                ragEvaluationCaseDraft.value = {
                    question: '', scenario: 'fact', expectedPoints: '', allowedSources: '', expectNoAnswer: false,
                };
                await selectRagEvaluationSet(evaluationSetId);
                await loadRagEvaluationSets();
                ragEvaluationMessage.value = '评测问题已加入。';
            } catch (error) {
                ragEvaluationMessage.value = error.message;
            } finally {
                ragEvaluationLoading.value = false;
            }
        }

        async function runRagEvaluation() {
            const evaluationSetId = selectedRagEvaluationSet.value?.evaluation_set_id;
            if (!evaluationSetId || ragEvaluationLoading.value) return;
            ragEvaluationLoading.value = true;
            ragEvaluationMessage.value = '正在按当前检索策略核验全部问题…';
            try {
                const res = await apiFetch(`${API_BASE}/rag/evaluation-sets/${encodeURIComponent(evaluationSetId)}/runs`, { method: 'POST' });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '评测运行失败');
                await selectRagEvaluationSet(evaluationSetId);
                ragEvaluationMessage.value = `评测完成：通过率 ${Math.round((data.run.metrics?.pass_rate || 0) * 100)}%，引用有效率 ${Math.round((data.run.metrics?.citation_validity_rate || 0) * 100)}%。`;
            } catch (error) {
                ragEvaluationMessage.value = error.message;
            } finally {
                ragEvaluationLoading.value = false;
            }
        }

        async function sendMessage() {
            const msg = input.value.trim();
            if (!msg || loading.value) return;
            const history = messages.value.map(({ role, content }) => ({ role, content }));
            input.value = '';
            messages.value.push({ role: 'user', content: msg });
            const smartEnabled = chatMode.value === 'smart';
            messages.value.push({
                role: 'assistant',
                content: '',
                sources: [],
                answerId: '',
                noAnswer: false,
                question: msg,
                feedback: '',
                feedbackOpen: false,
                feedbackReason: '',
                feedbackNote: '',
                pending: true,
                process: createAnalysisProcess(msg, useRag.value, smartEnabled),
            });
            const msgIdx = messages.value.length - 1;
            const assistantMessage = messages.value[msgIdx];
            const thinkingStartedAt = Date.now();
            loading.value = true;
            activeRetrievalStep.value = 0;
            const elapsedTimer = setInterval(() => {
                assistantMessage.process.elapsedSeconds = Math.max(1, Math.floor((Date.now() - thinkingStartedAt) / 1000));
            }, 250);
            scrollToBottom();

            try {
                const form = new FormData();
                form.append('message', msg);
                form.append('use_rag', String(useRag.value));
                form.append('history', JSON.stringify(history));

                const endpoint = smartEnabled ? 'smart-chat' : 'chat';
                const res = await apiFetch(`${API_BASE}/${endpoint}`, { method: 'POST', body: form });
                if (!res.ok || !res.body) {
                    const detail = await res.text();
                    throw new Error(detail || '服务暂时不可用');
                }
                const reader = res.body.getReader();
                const decoder = new TextDecoder();
                let fullContent = '';
                let sources = [];
                let streamBuffer = '';
                let answerStarted = false;

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    streamBuffer += decoder.decode(value, { stream: true });

                    while (!answerStarted) {
                        const lineEnd = streamBuffer.indexOf('\n');
                        if (lineEnd < 0) break;
                        const line = streamBuffer.slice(0, lineEnd);
                        streamBuffer = streamBuffer.slice(lineEnd + 1);

                        if (line === '__ANSWER__') {
                            answerStarted = true;
                            continue;
                        }
                        if (line.startsWith('__TRACE__:')) {
                            try {
                                updateAnalysisProcess(assistantMessage, JSON.parse(line.slice('__TRACE__:'.length)));
                            } catch (e) {
                                // 过程事件异常不应阻塞回答。
                            }
                            scrollToBottom();
                            continue;
                        }
                        if (line.startsWith('__SOURCES__:')) {
                            try {
                                const parsed = JSON.parse(line.slice('__SOURCES__:'.length));
                                sources = Array.isArray(parsed) ? parsed : (parsed.sources || []);
                                assistantMessage.sources = sources;
                            } catch (e) {
                                // 来源元数据异常不应阻塞正常回答。
                            }
                            continue;
                        }
                        if (line.startsWith('__ANSWER_META__:')) {
                            try {
                                const meta = JSON.parse(line.slice('__ANSWER_META__:'.length));
                                assistantMessage.answerId = meta.answer_id || '';
                                assistantMessage.noAnswer = Boolean(meta.no_answer);
                                assistantMessage.citationValidityRate = Number(meta.citation_validity_rate || 0);
                            } catch (e) {
                                // 回答元数据异常不应阻塞正文。
                            }
                            continue;
                        }
                        try {
                            const parsed = JSON.parse(line);
                            if (parsed.sources) {
                                sources = parsed.sources;
                                assistantMessage.sources = sources;
                                continue;
                            }
                        } catch (e) {
                            // 非协议行即视为旧服务返回的回答正文。
                        }
                        answerStarted = true;
                        streamBuffer = `${line}\n${streamBuffer}`;
                    }

                    if (!answerStarted || !streamBuffer) continue;
                    fullContent += streamBuffer;
                    streamBuffer = '';
                    messages.value[msgIdx].content = fullContent;
                    messages.value[msgIdx].pending = false;
                    scrollToBottom();
                }
                streamBuffer += decoder.decode();
                if (streamBuffer) {
                    fullContent += streamBuffer;
                    messages.value[msgIdx].content = fullContent;
                }
                messages.value[msgIdx].pending = false;
                messages.value[msgIdx].process.elapsedSeconds = Math.max(1, Math.floor((Date.now() - thinkingStartedAt) / 1000));
                const answerMissing = !messages.value[msgIdx].content.trim();
                if (answerMissing) {
                    messages.value[msgIdx].content = '未生成有效回答，请稍后重试。';
                }
                finishThinking(messages.value[msgIdx], sources.length, answerMissing);
            } catch (e) {
                messages.value[msgIdx].pending = false;
                messages.value[msgIdx].content = '请求失败: ' + e.message;
                messages.value[msgIdx].process.elapsedSeconds = Math.max(1, Math.floor((Date.now() - thinkingStartedAt) / 1000));
                finishThinking(messages.value[msgIdx], 0, true);
            } finally {
                clearInterval(elapsedTimer);
                activeRetrievalStep.value = retrievalSteps.length - 1;
                loading.value = false;
                scrollToBottom();
            }
        }

        function clearChat() {
            messages.value = [];
        }

        // ── 文档管理(通用) ──

        async function loadDocuments() {
            try {
                const res = await apiFetch(`${API_BASE}/documents`);
                const data = await res.json();
                if (data.success) documents.value = data.documents || [];
            } catch (e) { console.error('加载文档失败:', e); }
        }

        async function deleteDoc(filename, displayName = filename) {
            if (!confirm(`确定删除文档「${displayName}」吗？该来源对应的检索、图谱和知识引用也会被同步清理。`)) return false;
            try {
                const res = await apiFetch(`${API_BASE}/documents/${encodeURIComponent(filename)}`, { method: 'DELETE' });
                const data = await res.json();
                if (!res.ok || !data.success) throw new Error(data.detail || '服务端删除失败');
                await Promise.all([loadGroupedDocs(), loadKnowledgeAssets(), loadStats(), loadGraph()]);
                graphMsg.value = data.partial_success
                    ? `已删除「${displayName}」；${(data.sync_warnings || ['部分清理待重试']).join('、')}`
                    : `已删除「${displayName}」，知识图谱已同步更新`;
                return true;
            } catch (e) {
                alert('删除失败: ' + e.message);
                return false;
            }
        }

        async function loadStats() {
            try {
                const res = await apiFetch(`${API_BASE}/stats`);
                const data = await res.json();
                stats.value = data;
            } catch (e) { /* 忽略 */ }
        }

        // ── 知识图谱 ──

        async function loadGraph() {
            graphError.value = false;
            try {
                const res = await apiFetch(`${API_BASE}/knowledge-graph`);
                const data = await res.json();
                if (!res.ok || !data.success) throw new Error(data.detail || '图谱加载失败');
                graphData.value = normalizeGraphClient(data.graph || { nodes: [], edges: [] });
                renderGraphSoon();
            } catch (e) {
                graphData.value = { nodes: [], edges: [] };
                graphError.value = true;
                graphMsg.value = '图谱加载失败：' + e.message;
            }
        }

        async function buildGraph() {
            graphBuilding.value = true;
            graphMsg.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/knowledge-graph/build`, { method: 'POST' });
                const data = await res.json();
                if (data.success) {
                    graphMsg.value = `${data.message || '图谱重建已进入处理队列'}${data.job?.job_id ? ` · ${data.job.job_id}` : ''}`;
                    await loadProcessingJobs({ keepSelection: false, silent: true });
                } else {
                    alert(data.detail || '构建失败');
                }
            } catch (e) { alert('构建失败: ' + e.message); }
            finally { graphBuilding.value = false; }
        }

        function normalizeGraphClient(graph) {
            const nodes = (graph.nodes || []).map((node) => {
                const rawGroup = node.group || node.category || node.type || '项目知识';
                const displayGroup = categoryLabel(rawGroup);
                return ({
                id: String(node.id),
                label: node.label || node.name || '未命名节点',
                topic: node.topic || node.label || node.name || '未命名主题',
                group: displayGroup,
                group_code: rawGroup,
                type: categoryLabel(node.type || rawGroup || '知识节点'),
                title: node.title || node.content || '',
                source: node.source || '',
                location: node.location || '',
                tags: Array.isArray(node.tags) ? node.tags : [],
                size: Number(node.size || 18),
                x: Number.isFinite(Number(node.x)) ? Number(node.x) : undefined,
                y: Number.isFinite(Number(node.y)) ? Number(node.y) : undefined,
                locked: !!node.locked,
                origin: node.origin || (node.source ? 'generated' : 'manual'),
                manual_override: !!node.manual_override,
            });
            });
            const nodeIds = new Set(nodes.map((node) => node.id));
            const edges = (graph.edges || []).map((edge) => {
                const source = String(edge.source || edge.from || '');
                const target = String(edge.target || edge.to || '');
                return {
                    id: edge.id || `${source}__${target}__${edge.label || edge.relation || '关联'}`,
                    source,
                    target,
                    from: source,
                    to: target,
                    label: edge.label || edge.relation || '关联',
                    relation: edge.relation || edge.label || '关联',
                    width: Number(edge.width || 1),
                    confidence: Math.max(0, Math.min(1, Number(edge.confidence ?? (edge.auto_generated ? 0.65 : 0.9)))),
                    evidence: edge.evidence || '',
                    method: edge.method || (edge.auto_generated ? 'hybrid_auto' : 'manual'),
                    auto_generated: !!edge.auto_generated,
                    protected: edge.protected !== undefined ? !!edge.protected : !edge.auto_generated,
                };
            }).filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target) && edge.source !== edge.target);
            return { nodes, edges, updated_at: graph.updated_at };
        }

        function graphSnapshot() {
            return {
                nodes: (graphData.value.nodes || []).map((node) => ({
                    ...node,
                    x: Number.isFinite(Number(node.x)) ? Math.round(Number(node.x)) : undefined,
                    y: Number.isFinite(Number(node.y)) ? Math.round(Number(node.y)) : undefined,
                })),
                edges: (graphData.value.edges || []).map((edge) => ({
                    ...edge,
                    from: edge.source || edge.from,
                    to: edge.target || edge.to,
                    source: edge.source || edge.from,
                    target: edge.target || edge.to,
                })),
            };
        }

        async function saveGraphSnapshot() {
            graphSaving.value = true;
            try {
                const res = await apiFetch(`${API_BASE}/knowledge-graph`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(graphSnapshot()),
                });
                const data = await res.json();
                if (data.success) {
                    graphMsg.value = '图谱已保存';
                    graphData.value = normalizeGraphClient(data.graph || graphSnapshot());
                } else {
                    graphMsg.value = '图谱保存失败';
                }
            } catch (e) {
                graphMsg.value = '图谱保存失败：' + e.message;
            } finally {
                graphSaving.value = false;
            }
        }

        function scheduleGraphSave() {
            clearTimeout(graphSaveTimer);
            graphMsg.value = '正在保存图谱...';
            graphSaveTimer = setTimeout(() => saveGraphSnapshot(), 520);
        }

        function renderGraphSoon() {
            nextTick(() => requestAnimationFrame(renderKnowledgeGraph));
        }

        function updateGraphSelectionPresentation() {
            const container = graphCanvasRef.value;
            const d3 = window.d3;
            if (!container || !d3 || !container.querySelector('svg')) return;

            const selectedNodeId = selectedGraphNode.value?.id || '';
            const selectedEdgeId = selectedGraphEdge.value?.id || '';
            const endpointId = (value) => String(typeof value === 'object' ? value?.id : value || '');
            const connectedNodeIds = new Set();
            const links = d3.select(container).selectAll('.kg-link');

            if (selectedNodeId) {
                links.each((edge) => {
                    const source = endpointId(edge.source || edge.from);
                    const target = endpointId(edge.target || edge.to);
                    if (source === selectedNodeId || target === selectedNodeId) {
                        connectedNodeIds.add(source);
                        connectedNodeIds.add(target);
                    }
                });
            } else if (selectedEdgeId) {
                links.each((edge) => {
                    if (edge.id !== selectedEdgeId) return;
                    connectedNodeIds.add(endpointId(edge.source || edge.from));
                    connectedNodeIds.add(endpointId(edge.target || edge.to));
                });
            }

            links
                .classed('selected', (edge) => edge.id === selectedEdgeId)
                .classed('connected', (edge) => {
                    const source = endpointId(edge.source || edge.from);
                    const target = endpointId(edge.target || edge.to);
                    return !!selectedNodeId && (source === selectedNodeId || target === selectedNodeId);
                })
                .attr('stroke-width', (edge) => {
                    const currentEdge = findGraphEdge(edge.id) || edge;
                    return Math.max(1, Number(currentEdge.width || 1)) + (edge.id === selectedEdgeId ? 1 : 0);
                })
                .attr('opacity', (edge) => {
                    const currentEdge = findGraphEdge(edge.id) || edge;
                    const source = endpointId(edge.source || edge.from);
                    const target = endpointId(edge.target || edge.to);
                    if (selectedEdgeId) return edge.id === selectedEdgeId ? 1 : 0.2;
                    if (selectedNodeId) return source === selectedNodeId || target === selectedNodeId ? 1 : 0.18;
                    return currentEdge.auto_generated ? 0.68 : 0.9;
                });

            const nodes = d3.select(container).selectAll('.kg-node');
            nodes
                .classed('selected', (node) => node.id === selectedNodeId)
                .attr('opacity', (node) => {
                    if (!selectedNodeId && !selectedEdgeId) return 1;
                    return node.id === selectedNodeId || connectedNodeIds.has(node.id) ? 1 : 0.32;
                });
            nodes.select('circle')
                .attr('stroke', (node) => node.id === selectedNodeId ? '#ffffff' : 'rgba(255,255,255,0.55)')
                .attr('stroke-width', (node) => node.id === selectedNodeId ? 3 : 1.4);

            d3.select(container).selectAll('.kg-label')
                .classed('is-visible', function (node) {
                    return this.getAttribute('data-base-visible') === 'true'
                        || node.id === selectedNodeId
                        || connectedNodeIds.has(node.id);
                });
        }

        function findGraphNode(nodeId) {
            return (graphData.value.nodes || []).find((node) => node.id === nodeId);
        }

        function findGraphEdge(edgeId) {
            return (graphData.value.edges || []).find((edge) => edge.id === edgeId);
        }

        function selectGraphNode(node) {
            const sourceNode = typeof node === 'string' ? findGraphNode(node) : findGraphNode(node.id);
            if (!sourceNode) return;
            selectedGraphNode.value = { ...sourceNode };
            selectedGraphEdge.value = null;
            graphEditorMode.value = 'node';
            nodeDraft.value = {
                label: sourceNode.label || '',
                group: sourceNode.group || '项目知识',
                type: sourceNode.type || '知识节点',
                title: sourceNode.title || '',
                size: sourceNode.size || 18,
            };
            edgeDraft.value.source = sourceNode.id;
            if (!edgeDraft.value.target || edgeDraft.value.target === sourceNode.id) {
                const target = (graphData.value.nodes || []).find((item) => item.id !== sourceNode.id);
                edgeDraft.value.target = target ? target.id : '';
            }
            updateGraphSelectionPresentation();
        }

        function selectGraphEdge(edge) {
            const sourceEdge = typeof edge === 'string' ? findGraphEdge(edge) : findGraphEdge(edge.id);
            if (!sourceEdge) return;
            selectedGraphEdge.value = { ...sourceEdge };
            selectedGraphNode.value = null;
            graphEditorMode.value = 'edge';
            edgeDraft.value = {
                source: sourceEdge.source || sourceEdge.from,
                target: sourceEdge.target || sourceEdge.to,
                label: sourceEdge.label || '关联',
                width: sourceEdge.width || 2,
                confidence: Number(sourceEdge.confidence ?? (sourceEdge.auto_generated ? 0.65 : 0.9)),
                evidence: sourceEdge.evidence || '',
                protected: sourceEdge.auto_generated ? true : sourceEdge.protected !== false,
            };
            updateGraphSelectionPresentation();
        }

        function clearGraphSelection() {
            if (!selectedGraphNode.value && !selectedGraphEdge.value) return;
            selectedGraphNode.value = null;
            selectedGraphEdge.value = null;
            updateGraphSelectionPresentation();
        }

        function applyNodeDraft() {
            if (!selectedGraphNode.value) return;
            const node = findGraphNode(selectedGraphNode.value.id);
            if (!node) return;
            Object.assign(node, {
                label: nodeDraft.value.label || '未命名节点',
                topic: nodeDraft.value.label || '未命名主题',
                group: nodeDraft.value.group || '项目知识',
                type: nodeDraft.value.type || '知识节点',
                title: nodeDraft.value.title || '',
                size: Number(nodeDraft.value.size || 18),
                manual_override: true,
            });
            selectedGraphNode.value = { ...node };
            scheduleGraphSave();
            renderGraphSoon();
        }

        function addGraphNode() {
            const id = `node_${Date.now().toString(36)}`;
            const node = {
                id,
                label: newNodeDraft.value.label.trim() || '新知识节点',
                topic: newNodeDraft.value.label.trim() || '新知识主题',
                group: newNodeDraft.value.group.trim() || '项目知识',
                type: newNodeDraft.value.type.trim() || '知识节点',
                title: newNodeDraft.value.title.trim(),
                size: 18,
                origin: 'manual',
                manual_override: true,
            };
            graphData.value.nodes.push(node);
            newNodeDraft.value = { label: '', group: node.group, type: node.type, title: '' };
            selectGraphNode(node);
            scheduleGraphSave();
            renderGraphSoon();
        }

        function deleteGraphNode() {
            if (!selectedGraphNode.value) return;
            const nodeId = selectedGraphNode.value.id;
            graphData.value.nodes = graphData.value.nodes.filter((node) => node.id !== nodeId);
            graphData.value.edges = graphData.value.edges.filter((edge) => (edge.source || edge.from) !== nodeId && (edge.target || edge.to) !== nodeId);
            clearGraphSelection();
            scheduleGraphSave();
            renderGraphSoon();
        }

        function applyEdgeDraft() {
            if (!selectedGraphEdge.value) return;
            const edge = findGraphEdge(selectedGraphEdge.value.id);
            if (!edge || edgeDraft.value.source === edgeDraft.value.target) return;
            const topologyChanged = (edge.source || edge.from) !== edgeDraft.value.source
                || (edge.target || edge.to) !== edgeDraft.value.target;
            Object.assign(edge, {
                source: edgeDraft.value.source,
                target: edgeDraft.value.target,
                from: edgeDraft.value.source,
                to: edgeDraft.value.target,
                label: edgeDraft.value.label || '关联',
                relation: edgeDraft.value.label || '关联',
                width: Number(edgeDraft.value.width || 2),
                confidence: Number(edgeDraft.value.confidence || 0.9),
                evidence: edgeDraft.value.evidence || '人工确认的关系',
                method: 'manual',
                auto_generated: false,
                protected: !!edgeDraft.value.protected,
            });
            selectedGraphEdge.value = { ...edge };
            scheduleGraphSave();
            if (topologyChanged) {
                renderGraphSoon();
            } else {
                updateGraphSelectionPresentation();
            }
        }

        function addGraphEdge() {
            if (!edgeDraft.value.source || !edgeDraft.value.target || edgeDraft.value.source === edgeDraft.value.target) {
                graphMsg.value = '请选择两个不同节点';
                return;
            }
            const edge = {
                id: `edge_${Date.now().toString(36)}`,
                source: edgeDraft.value.source,
                target: edgeDraft.value.target,
                from: edgeDraft.value.source,
                to: edgeDraft.value.target,
                label: edgeDraft.value.label || '关联',
                relation: edgeDraft.value.label || '关联',
                width: Number(edgeDraft.value.width || 2),
                confidence: Number(edgeDraft.value.confidence || 0.9),
                evidence: edgeDraft.value.evidence || '人工新增关系',
                method: 'manual',
                auto_generated: false,
                protected: edgeDraft.value.protected !== false,
            };
            graphData.value.edges.push(edge);
            selectGraphEdge(edge);
            scheduleGraphSave();
            renderGraphSoon();
        }

        function deleteGraphEdge() {
            if (!selectedGraphEdge.value) return;
            graphData.value.edges = graphData.value.edges.filter((edge) => edge.id !== selectedGraphEdge.value.id);
            selectedGraphEdge.value = null;
            graphEditorMode.value = 'edge';
            scheduleGraphSave();
            renderGraphSoon();
        }

        function resetGraphView() {
            graphSearch.value = '';
            graphGroupFilter.value = '全部';
            clearGraphSelection();
            nextTick(() => {
                if (graphFitToContent) graphFitToContent(true);
                else renderGraphSoon();
            });
        }

        function renderKnowledgeGraph() {
            const container = graphCanvasRef.value;
            if (!container || tab.value !== 'graph') return;
            container.innerHTML = '';
            const d3 = window.d3;
            if (!d3) {
                container.innerHTML = '<div class="graph-engine-state">图谱引擎加载中，请稍后刷新。</div>';
                return;
            }
            if (graphSimulation) graphSimulation.stop();

            const width = Math.max(container.clientWidth || 900, 720);
            const height = Math.max(container.clientHeight || 560, 520);
            const nodes = filteredGraph.value.nodes.map((node) => ({
                ...node,
                radius: Math.max(6, Math.min(28, Number(node.size || 18))),
            }));
            const nodeIds = new Set(nodes.map((node) => node.id));
            const edges = filteredGraph.value.edges
                .map((edge) => ({ ...edge, source: edge.source || edge.from, target: edge.target || edge.to }))
                .filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target));

            if (!nodes.length) {
                container.innerHTML = '<div class="graph-engine-state">没有匹配节点，换个关键词或新增一个节点。</div>';
                return;
            }

            const groupNames = [...new Set(nodes.map((node) => node.group || '项目知识'))];
            const colorOf = (group) => graphColorAt(Math.max(0, groupNames.indexOf(group || '项目知识')));
            const degreeByNode = new Map(nodes.map((item) => [item.id, 0]));
            edges.forEach((edge) => {
                degreeByNode.set(edge.source, (degreeByNode.get(edge.source) || 0) + 1);
                degreeByNode.set(edge.target, (degreeByNode.get(edge.target) || 0) + 1);
            });
            const smartLabelLimit = nodes.length <= 18
                ? nodes.length
                : Math.min(32, Math.max(10, Math.ceil(nodes.length * 0.18)));
            const priorityNodeIds = new Set(
                [...nodes]
                    .sort((a, b) => (degreeByNode.get(b.id) || 0) - (degreeByNode.get(a.id) || 0) || b.radius - a.radius)
                    .slice(0, smartLabelLimit)
                    .map((item) => item.id)
            );
            const compactLabel = (value, maxLength = 16) => {
                const text = String(value || '未命名主题').trim();
                return text.length > maxLength ? `${text.slice(0, maxLength)}…` : text;
            };
            const shouldShowBaseLabel = (item) => graphLabelMode.value === 'all'
                || !!graphSearch.value.trim()
                || priorityNodeIds.has(item.id);
            const shouldShowLabel = (item) => shouldShowBaseLabel(item)
                || selectedGraphNode.value?.id === item.id;

            const svg = d3.select(container)
                .append('svg')
                .attr('viewBox', [0, 0, width, height])
                .attr('role', 'img')
                .attr('aria-label', '可编辑知识图谱');

            const viewport = svg.append('g');
            const zoom = d3.zoom()
                .scaleExtent([0.08, 3])
                .on('zoom', (event) => {
                    viewport.attr('transform', event.transform);
                    graphZoomLabel.value = `${Math.round(event.transform.k * 100)}%`;
                });
            svg.call(zoom).on('dblclick.zoom', null);
            svg.on('click', () => clearGraphSelection());

            viewport.append('g')
                .attr('class', 'kg-links')
                .selectAll('line')
                .data(edges)
                .enter()
                .append('line')
                .attr('class', (edge) => `kg-link ${selectedGraphEdge.value?.id === edge.id ? 'selected' : ''}`)
                .attr('stroke-width', (edge) => Math.max(1, Number(edge.width || 1)))
                .attr('stroke-dasharray', (edge) => edge.auto_generated ? '6 5' : null)
                .attr('opacity', (edge) => edge.auto_generated ? 0.68 : 0.9)
                .on('click', (event, edge) => {
                    event.stopPropagation();
                    selectGraphEdge(edge);
                });

            const link = viewport.selectAll('.kg-link');
            const node = viewport.append('g')
                .attr('class', 'kg-nodes')
                .selectAll('g')
                .data(nodes)
                .enter()
                .append('g')
                .attr('class', (item) => `kg-node ${selectedGraphNode.value?.id === item.id ? 'selected' : ''}`)
                .on('click', (event, item) => {
                    event.stopPropagation();
                    selectGraphNode(item);
                })
                .on('mouseenter', (event, item) => {
                    d3.select(event.currentTarget).classed('hovered', true).raise();
                    label.classed('is-visible', (candidate) => shouldShowLabel(candidate) || candidate.id === item.id);
                })
                .on('mouseleave', (event) => {
                    d3.select(event.currentTarget).classed('hovered', false);
                    label.classed('is-visible', (candidate) => shouldShowLabel(candidate));
                })
                .call(d3.drag()
                    .on('start', (event, item) => {
                        if (!event.active) graphSimulation.alphaTarget(0.25).restart();
                        item.fx = item.x;
                        item.fy = item.y;
                    })
                    .on('drag', (event, item) => {
                        item.fx = event.x;
                        item.fy = event.y;
                    })
                    .on('end', (event, item) => {
                        if (!event.active) graphSimulation.alphaTarget(0);
                        const sourceNode = findGraphNode(item.id);
                        if (sourceNode) {
                            sourceNode.x = item.x;
                            sourceNode.y = item.y;
                            sourceNode.locked = true;
                            scheduleGraphSave();
                        }
                    }));

            node.append('circle')
                .attr('r', (item) => item.radius)
                .attr('fill', (item) => colorOf(item.group))
                .attr('stroke', (item) => selectedGraphNode.value?.id === item.id ? '#ffffff' : 'rgba(255,255,255,0.55)')
                .attr('stroke-width', (item) => selectedGraphNode.value?.id === item.id ? 3 : 1.4);

            const label = node.append('text')
                .attr('dy', (item) => -(item.radius + 10))
                .attr('text-anchor', 'middle')
                .text((item) => compactLabel(item.topic || item.label))
                .attr('data-base-visible', (item) => shouldShowBaseLabel(item) ? 'true' : 'false')
                .attr('class', (item) => `kg-label ${shouldShowLabel(item) ? 'is-visible' : ''}`);

            node.append('title').text((item) => [
                item.topic || item.label,
                item.source ? `来源：${item.source}` : '',
                item.location || '',
                item.title || '',
            ].filter(Boolean).join('\n'));

            graphSimulation = d3.forceSimulation(nodes)
                .force('link', d3.forceLink(edges).id((item) => item.id).distance(138).strength(0.44))
                .force('charge', d3.forceManyBody().strength(-310))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collision', d3.forceCollide((item) => item.radius + 38))
                .alpha(0.9);

            nodes.forEach((item) => {
                if (Number.isFinite(Number(item.x))) item.x = Number(item.x);
                if (Number.isFinite(Number(item.y))) item.y = Number(item.y);
                if (item.locked && Number.isFinite(item.x) && Number.isFinite(item.y)) {
                    item.fx = item.x;
                    item.fy = item.y;
                }
            });

            const updateGraphPositions = () => {
                link
                    .attr('x1', (edge) => edge.source.x)
                    .attr('y1', (edge) => edge.source.y)
                    .attr('x2', (edge) => edge.target.x)
                    .attr('y2', (edge) => edge.target.y);
                node.attr('transform', (item) => `translate(${item.x},${item.y})`);
            };

            graphSimulation.stop();
            const settleTicks = Math.min(150, 70 + Math.ceil(nodes.length * 0.4));
            for (let index = 0; index < settleTicks; index += 1) graphSimulation.tick();
            updateGraphPositions();

            graphFitToContent = (animate = false) => {
                const positioned = nodes.filter((item) => Number.isFinite(item.x) && Number.isFinite(item.y));
                if (!positioned.length) return;
                const labelAllowanceX = 92;
                const labelAllowanceY = 46;
                const minX = d3.min(positioned, (item) => item.x - item.radius - labelAllowanceX);
                const maxX = d3.max(positioned, (item) => item.x + item.radius + labelAllowanceX);
                const minY = d3.min(positioned, (item) => item.y - item.radius - labelAllowanceY);
                const maxY = d3.max(positioned, (item) => item.y + item.radius + labelAllowanceY);
                const boundsWidth = Math.max(1, maxX - minX);
                const boundsHeight = Math.max(1, maxY - minY);
                const padding = Math.min(72, Math.max(28, Math.min(width, height) * 0.08));
                const scale = Math.max(0.08, Math.min(1, Math.min(
                    (width - padding * 2) / boundsWidth,
                    (height - padding * 2) / boundsHeight,
                )));
                const centerX = (minX + maxX) / 2;
                const centerY = (minY + maxY) / 2;
                const transform = d3.zoomIdentity
                    .translate(width / 2 - scale * centerX, height / 2 - scale * centerY)
                    .scale(scale);
                const target = animate ? svg.transition().duration(260) : svg;
                target.call(zoom.transform, transform);
            };
            graphFitToContent(false);

            graphSimulation.on('tick', updateGraphPositions);
        }

        // ── 飞书 ──

        async function pushDaily() {
            feishuLoading.value = true;
            feishuMsg.value = '';
            try {
                const form = new FormData();
                if (pushChatId.value.trim()) form.append('chat_id', pushChatId.value.trim());
                const res = await apiFetch(`${API_BASE}/feishu/push/knowledge`, { method: 'POST', body: form });
                const data = await res.json();
                feishuMsg.value = data.message || (data.success ? '推送成功' : '推送失败');
            } catch (e) {
                feishuMsg.value = '推送失败: ' + e.message;
            } finally {
                feishuLoading.value = false;
            }
        }

        async function saveFeishuSettings() {
            feishuLoading.value = true;
            feishuMsg.value = '';
            try {
                const form = new FormData();
                form.append('feishuAppId', feishuAppId.value.trim());
                if (feishuAppSecret.value.trim()) form.append('feishuAppSecret', feishuAppSecret.value.trim());
                form.append('feishuDefaultChatId', feishuDefaultChatId.value.trim());
                const res = await apiFetch(`${API_BASE}/settings/feishu`, { method: 'POST', body: form });
                const data = await res.json();
                feishuMsg.value = data.success ? '成功：' + data.message : '失败：' + (data.detail || '保存失败');
                if (data.success) {
                    feishuConfigured.value = true;
                    feishuShowCredentialForm.value = false;
                    await loadFeishuWorkspace();
                }
            } catch (e) { feishuMsg.value = '失败：保存失败: ' + e.message; }
            finally { feishuLoading.value = false; }
        }

        function applyFeishuWorkspace(data) {
            if (!data?.success) return;
            feishuWorkspace.value = data;
            feishuConfigured.value = !!data.connection?.configured;
            const candidateIds = new Set((data.candidates || []).map((candidate) => candidate.candidate_id));
            feishuSelectedCandidateIds.value = feishuSelectedCandidateIds.value.filter((id) => candidateIds.has(id));
            const publishedAssetIds = new Set((data.assets || []).filter((asset) => asset.status === 'published').map((asset) => asset.asset_id));
            feishuSelectedAssetIds.value = feishuSelectedAssetIds.value.filter((id) => publishedAssetIds.has(id));
            if (!feishuSelectedGroupId.value && data.groups?.length) {
                feishuSelectedGroupId.value = data.groups[0].chat_id;
            }
            if (feishuSelectedGroupId.value && !data.groups?.some((group) => group.chat_id === feishuSelectedGroupId.value)) {
                feishuSelectedGroupId.value = data.groups?.[0]?.chat_id || '';
            }
        }

        function toLocalIsoString(date) {
            // 输出本地时区的“无时区”ISO 字符串（不带 Z），避免 toISOString() 转成 UTC
            // 导致与后端按本地时间存储的消息时间戳产生 8 小时偏差、把今天的消息误过滤掉。
            const pad = (n) => String(n).padStart(2, '0');
            return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`
                + `T${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
        }

        function feishuContentDateBounds() {
            const days = Number(feishuDateRange.value || 0);
            if (!days) return { dateFrom: '', dateTo: '' };
            const dateTo = new Date();
            const dateFrom = new Date();
            dateFrom.setHours(0, 0, 0, 0);
            dateFrom.setDate(dateFrom.getDate() - Math.max(0, days - 1));
            return { dateFrom: toLocalIsoString(dateFrom), dateTo: toLocalIsoString(dateTo) };
        }

        async function loadFeishuContentData({ resetPage = false } = {}) {
            if (resetPage) feishuContentPage.value = 1;
            const requestId = ++feishuContentRequestId;
            feishuContentLoading.value = true;
            try {
                const params = new URLSearchParams({
                    page: String(feishuContentPage.value),
                    page_size: String(feishuContentPageSize),
                });
                if (feishuHistoryQuery.value.trim()) params.set('keyword', feishuHistoryQuery.value.trim());

                let endpoint = `${API_BASE}/feishu/content/assets`;
                if (feishuContentSection.value === 'assets') {
                    params.set('statuses', 'published');
                } else if (feishuContentSection.value === 'messages') {
                    endpoint = `${API_BASE}/feishu/content/messages`;
                    const { dateFrom, dateTo } = feishuContentDateBounds();
                    if (feishuContentGroupId.value) params.set('chat_id', feishuContentGroupId.value);
                    if (feishuContentStatus.value) params.set('statuses', feishuContentStatus.value);
                    params.set('exclude_statuses', 'reverted,excluded,not_collected');
                    if (feishuMessageType.value) params.set('message_type', feishuMessageType.value);
                    if (dateFrom) params.set('date_from', dateFrom);
                    if (dateTo) params.set('date_to', dateTo);
                } else if (feishuHistoryType.value === 'assets') {
                    params.set('statuses', 'reverted');
                } else {
                    endpoint = `${API_BASE}/feishu/content/messages`;
                    const { dateFrom, dateTo } = feishuContentDateBounds();
                    params.set('statuses', 'reverted,excluded');
                    if (feishuContentGroupId.value) params.set('chat_id', feishuContentGroupId.value);
                    if (feishuMessageType.value) params.set('message_type', feishuMessageType.value);
                    if (dateFrom) params.set('date_from', dateFrom);
                    if (dateTo) params.set('date_to', dateTo);
                }

                const res = await apiFetch(`${endpoint}?${params.toString()}`);
                const data = await res.json();
                if (!res.ok || !data.success) throw new Error(data.detail || '内容列表加载失败');
                if (requestId !== feishuContentRequestId) return;
                feishuContentPage.value = Number(data.page || 1);
                feishuContentPages.value = Number(data.pages || 1);
                feishuContentTotal.value = Number(data.total || 0);
                if (endpoint.endsWith('/messages')) {
                    feishuContentMessages.value = data.items || [];
                    feishuContentAssets.value = [];
                } else {
                    feishuContentAssets.value = data.items || [];
                    feishuContentMessages.value = [];
                    const availableIds = new Set(feishuPublishedAssets.value.map((asset) => asset.asset_id));
                    feishuSelectedAssetIds.value = feishuSelectedAssetIds.value.filter((id) => availableIds.has(id));
                }
            } catch (error) {
                if (requestId === feishuContentRequestId) feishuMsg.value = '内容中心加载失败：' + error.message;
            } finally {
                if (requestId === feishuContentRequestId) feishuContentLoading.value = false;
            }
        }

        function scheduleFeishuContentReload() {
            if (feishuContentReloadTimer) clearTimeout(feishuContentReloadTimer);
            feishuContentReloadTimer = setTimeout(() => loadFeishuContentData({ resetPage: true }), 240);
        }

        function selectFeishuContentSection(section) {
            feishuContentSection.value = section;
            feishuContentStatus.value = '';
            feishuContentPage.value = 1;
        }

        function selectFeishuHistoryType(type) {
            feishuHistoryType.value = type;
            feishuContentPage.value = 1;
        }

        function changeFeishuContentPage(page) {
            const target = Math.max(1, Math.min(Number(page || 1), feishuContentPages.value));
            if (target === feishuContentPage.value) return;
            feishuContentPage.value = target;
            loadFeishuContentData();
        }

        async function loadFeishuWorkspace() {
            feishuWorkspaceLoading.value = true;
            try {
                const res = await apiFetch(`${API_BASE}/feishu/workspace`);
                const data = await res.json();
                if (!data.success) throw new Error(data.detail || '飞书工作台加载失败');
                applyFeishuWorkspace(data);
                if (tab.value === 'feishu' && feishuView.value === 'content') {
                    await loadFeishuContentData();
                }
            } catch (e) {
                feishuMsg.value = '工作台加载失败：' + e.message;
            } finally {
                feishuWorkspaceLoading.value = false;
            }
        }

        function connectFeishuWorkspaceEvents() {
            if (!window.EventSource) {
                feishuLiveSyncState.value = 'unsupported';
                return;
            }
            if (feishuEventSource) feishuEventSource.close();
            feishuLiveSyncState.value = 'connecting';
            feishuEventSource = new EventSource(`${API_BASE}/feishu/events`, { withCredentials: true });
            feishuEventSource.addEventListener('workspace', (event) => {
                try {
                    applyFeishuWorkspace(JSON.parse(event.data));
                    feishuLiveSyncState.value = 'connected';
                    if (tab.value === 'feishu' && feishuView.value === 'content') {
                        scheduleFeishuContentReload();
                    }
                } catch (error) {
                    console.error('飞书实时消息解析失败:', error);
                }
            });
            feishuEventSource.onopen = () => { feishuLiveSyncState.value = 'connected'; };
            feishuEventSource.onerror = () => { feishuLiveSyncState.value = 'reconnecting'; };
        }

        function feishuLiveSyncLabel() {
            return {
                connected: '实时同步中',
                connecting: '正在连接实时消息',
                reconnecting: '实时消息重连中',
                unsupported: '浏览器不支持实时同步',
            }[feishuLiveSyncState.value] || '实时同步未连接';
        }

        async function startFeishuEventConnection() {
            feishuWorkspaceLoading.value = true;
            feishuMsg.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/feishu/connection/start`, { method: 'POST' });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || data.message || '长连接启动失败');
                const connecting = data.status?.running && data.status?.state === 'connecting';
                feishuMsg.value = data.success ? '成功：' + data.message
                    : connecting ? '连接中：' + data.message
                    : '连接失败：' + (data.status?.last_error || data.message || '事件通道未建立');
                await loadFeishuWorkspace();
            } catch (e) {
                feishuMsg.value = '失败：' + e.message;
            } finally {
                feishuWorkspaceLoading.value = false;
            }
        }

        function feishuTransportLabel() {
            if (feishuEventConnected.value) return '事件通道已连接';
            if (feishuTransport.value.state === 'connecting') return '事件通道连接中';
            if (feishuTransport.value.state === 'stopping') return '事件通道正在停止';
            if (feishuTransport.value.state === 'error') return '事件通道连接失败';
            if (!feishuConnection.value.configured) return '等待配置凭证';
            return '事件通道未连接';
        }

        function feishuActionLabel(action) {
            return {
                processing: '正在处理消息',
                replied: '已回复消息',
                resignation_guide: '已发送离职指引',
                reply_failed: '消息已收到，但机器人回复失败',
                ignored_not_mentioned: '未授权采集，且未触发回复',
                collected_no_reply: '消息已归档，未触发回复',
                challenge_verified: 'Webhook 地址验证通过',
                group_connected: '群聊已接入并完成历史同步',
                group_connect_failed: '群聊接入失败',
            }[action] || (action ? '已收到事件' : '尚未收到事件');
        }

        async function discoverFeishuChats() {
            feishuDiscoveringChats.value = true;
            feishuDiscoveryMsg.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/feishu/chats`);
                const data = await res.json();
                if (!res.ok || !data.success) throw new Error(data.detail || data.message || '群聊发现失败');
                feishuDiscoveredChats.value = data.chats || [];
                feishuDiscoveryMsg.value = data.message || `已发现 ${feishuDiscoveredChats.value.length} 个群聊`;
            } catch (error) {
                feishuDiscoveryMsg.value = '拉取失败：' + error.message;
            } finally {
                feishuDiscoveringChats.value = false;
            }
        }

        function selectFeishuAvailableChat(chat) {
            const existing = feishuGroups.value.find((group) => group.chat_id === chat.chat_id);
            if (existing) {
                selectFeishuGroup(existing);
                return;
            }
            feishuSelectedGroupId.value = '';
            feishuGroupDraft.value = {
                chat_id: chat.chat_id,
                name: chat.name || '',
                description: chat.description || '',
                avatar: chat.avatar || '',
                chat_type: 'group',
                collection_mode: chat.external ? 'archive_only' : 'auto',
                default_category: 'proj_communication',
                retention_days: 90,
                bot_enabled: true,
                external: !!chat.external,
                confidence_threshold: 0.85,
                aggregation_window_minutes: 30,
            };
            feishuView.value = 'sources';
        }

        function selectFeishuGroup(group, view = 'sources') {
            feishuSelectedGroupId.value = group.chat_id;
            feishuGroupDraft.value = {
                chat_id: group.chat_id,
                name: group.name || '',
                collection_mode: group.collection_mode || (group.external ? 'archive_only' : 'auto'),
                default_category: group.default_category || 'proj_communication',
                retention_days: Number(group.retention_days || 90),
                bot_enabled: group.bot_enabled !== false,
                description: group.description || '',
                avatar: group.avatar || '',
                chat_type: group.chat_type || 'group',
                external: !!group.external,
                confidence_threshold: Number(group.confidence_threshold || 0.85),
                aggregation_window_minutes: Number(group.aggregation_window_minutes || 30),
            };
            feishuPolicyPreview.value = null;
            feishuView.value = view;
        }

        function startFeishuGroupDraft() {
            feishuGroupDraft.value = {
                chat_id: feishuConnection.value.default_chat_id || '',
                name: '',
                description: '',
                avatar: '',
                chat_type: 'group',
                collection_mode: 'auto',
                default_category: 'proj_communication',
                retention_days: 90,
                bot_enabled: true,
                external: false,
                confidence_threshold: 0.85,
                aggregation_window_minutes: 30,
            };
            feishuSelectedGroupId.value = '';
            feishuPolicyPreview.value = null;
            feishuView.value = 'sources';
        }

        async function saveFeishuGroup() {
            feishuSavingGroup.value = true;
            feishuMsg.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/feishu/groups`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(feishuGroupDraft.value),
                });
                const data = await res.json();
                if (!data.success) throw new Error(data.detail || '群采集规则保存失败');
                feishuMsg.value = data.message || '群采集规则已保存';
                feishuSelectedGroupId.value = data.group.chat_id;
                await loadFeishuWorkspace();
                await discoverFeishuChats();
                selectFeishuGroup(data.group);
            } catch (e) {
                feishuMsg.value = '保存失败：' + e.message;
            } finally {
                feishuSavingGroup.value = false;
            }
        }

        async function previewFeishuPolicy() {
            if (!feishuGroupDraft.value.chat_id) {
                feishuMsg.value = '请先选择一个采集源';
                return;
            }
            feishuPolicyPreviewLoading.value = true;
            feishuPolicyPreview.value = null;
            try {
                const res = await apiFetch(`${API_BASE}/feishu/policies/preview`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ...feishuGroupDraft.value, limit: 100 }),
                });
                const data = await res.json();
                if (!res.ok || !data.success) throw new Error(data.detail || '规则预览失败');
                feishuPolicyPreview.value = data.preview;
            } catch (error) {
                feishuMsg.value = '预览失败：' + error.message;
            } finally {
                feishuPolicyPreviewLoading.value = false;
            }
        }

        async function checkFeishuPermissions() {
            feishuPermissionLoading.value = true;
            feishuPermissionResult.value = null;
            try {
                const res = await apiFetch(`${API_BASE}/feishu/permissions/check`, { method: 'POST' });
                const data = await res.json();
                if (!res.ok || !data.success) throw new Error(data.detail || '权限检测失败');
                feishuPermissionResult.value = data;
                feishuMsg.value = data.verified ? '权限检测完成：核心接收能力已验证' : '权限检测完成：仍有能力待验证';
                await loadFeishuWorkspace();
            } catch (error) {
                feishuMsg.value = '权限检测失败：' + error.message;
            } finally {
                feishuPermissionLoading.value = false;
            }
        }

        function toggleFeishuCandidate(candidateId) {
            if (feishuSelectedCandidateIds.value.includes(candidateId)) {
                feishuSelectedCandidateIds.value = feishuSelectedCandidateIds.value.filter((id) => id !== candidateId);
            } else {
                feishuSelectedCandidateIds.value = [...feishuSelectedCandidateIds.value, candidateId];
            }
        }

        function toggleAllFeishuCandidates() {
            feishuSelectedCandidateIds.value = feishuAllPendingSelected.value
                ? []
                : feishuPendingCandidates.value.map((candidate) => candidate.candidate_id);
        }

        async function batchFeishuCandidates(action) {
            if (!feishuSelectedCandidateIds.value.length) {
                feishuMsg.value = '请至少选择一个知识候选';
                return;
            }
            feishuBatchLoading.value = true;
            try {
                const selectedCount = feishuSelectedCandidateIds.value.length;
                const res = await apiFetch(`${API_BASE}/feishu/candidates/batch-actions`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        action,
                        candidate_ids: feishuSelectedCandidateIds.value,
                        category: action === 'category' ? feishuBatchCategory.value : '',
                    }),
                });
                const data = await res.json();
                if (!res.ok || !data.success) throw new Error(data.detail || data.message || '批量处理提交失败');
                const acceptedJob = data.job;
                feishuSelectedCandidateIds.value = [];
                processingJobStatusFilter.value = 'active';
                processingJobTypeFilter.value = 'feishu_candidate_batch';
                processingJobKeyword.value = '';
                processingJobMessage.value = data.message || `已提交 ${selectedCount} 项候选`;
                tab.value = 'jobs';
                await loadProcessingJobs({ keepSelection: false });
                if (acceptedJob?.job_id) {
                    await openProcessingJob(acceptedJob, { silent: true });
                    trackFeishuCandidateBatchJob(acceptedJob.job_id, action);
                }
            } catch (error) {
                feishuMsg.value = '批量处理提交失败：' + error.message;
            } finally {
                feishuBatchLoading.value = false;
            }
        }

        async function trackFeishuCandidateBatchJob(jobId, action) {
            let consecutiveErrors = 0;
            for (let attempt = 0; attempt < 300; attempt += 1) {
                await new Promise(resolve => setTimeout(resolve, 1000));
                try {
                    const res = await apiFetch(`${API_BASE}/processing/jobs/${encodeURIComponent(jobId)}`);
                    const data = await res.json().catch(() => ({}));
                    if (!res.ok || !data.success) throw new Error(data.detail || '任务状态读取失败');
                    consecutiveErrors = 0;
                    const job = data.job;
                    if (selectedProcessingJob.value?.job_id === jobId) selectedProcessingJob.value = job;
                    if (!['succeeded', 'failed', 'cancelled'].includes(job.status)) continue;
                    processingJobStatusFilter.value = '';
                    const result = job.result || {};
                    if (job.status === 'succeeded') {
                        processingJobMessage.value = result.message || '飞书候选批量处理完成';
                        const latestAsset = [...(result.succeeded || [])].reverse()
                            .find(item => item.result?.asset)?.result?.asset;
                        feishuUndoAsset.value = latestAsset || null;
                    } else if (job.status === 'cancelled') {
                        processingJobMessage.value = `批量任务已取消，取消前完成 ${result.processed || 0} 项`;
                    } else {
                        processingJobMessage.value = `批量处理失败：${job.last_error || '请查看尝试记录后重试'}`;
                    }
                    await loadFeishuWorkspace();
                    if (['approve', 'retry'].includes(action) && Number(result.succeeded_count || 0) > 0) {
                        await Promise.all([loadDocuments(), loadStats(), loadGraph()]);
                    }
                    await loadProcessingJobs({ silent: true });
                    return;
                } catch (error) {
                    consecutiveErrors += 1;
                    if (consecutiveErrors >= 5) {
                        processingJobMessage.value = '批量任务状态暂时无法读取，可稍后刷新任务中心';
                        return;
                    }
                }
            }
            processingJobMessage.value = '批量任务仍在处理中，可稍后在任务中心继续查看';
        }

        function openFeishuRevokeConfirm(assets) {
            const targets = (Array.isArray(assets) ? assets : [assets]).filter(asset => asset?.asset_id);
            if (!targets.length) {
                feishuMsg.value = '请至少选择一个已入库知识资产';
                return;
            }
            feishuRevokeTargets.value = targets;
            feishuRevokeReason.value = '';
            feishuRevokeConfirmOpen.value = true;
        }

        function closeFeishuRevokeConfirm() {
            if (feishuBatchLoading.value) return;
            feishuRevokeConfirmOpen.value = false;
            feishuRevokeReason.value = '';
            feishuRevokeTargets.value = [];
        }

        function revertFeishuAsset(asset) {
            openFeishuRevokeConfirm(asset);
        }

        async function confirmRevertFeishuAssets() {
            const targets = feishuRevokeTargets.value.filter(asset => asset?.asset_id);
            const reason = feishuRevokeReason.value.trim();
            if (!targets.length) {
                feishuMsg.value = '撤销目标已失效，请刷新后重试';
                closeFeishuRevokeConfirm();
                return;
            }
            if (!reason) {
                feishuMsg.value = '撤销失败：必须填写真实撤销原因';
                return;
            }
            feishuBatchLoading.value = true;
            try {
                const isBatch = targets.length > 1;
                const endpoint = isBatch
                    ? `${API_BASE}/feishu/assets/batch-revert`
                    : `${API_BASE}/feishu/assets/${encodeURIComponent(targets[0].asset_id)}/revert`;
                const payload = isBatch
                    ? { asset_ids: targets.map(asset => asset.asset_id), reason }
                    : { reason };
                const res = await apiFetch(endpoint, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
                });
                const data = await res.json().catch(() => ({}));
                const succeeded = Array.isArray(data.succeeded) ? data.succeeded : [];
                if (!res.ok || (!data.success && (!isBatch || !succeeded.length))) {
                    throw new Error(data.detail || data.message || '撤销失败');
                }
                const partialResults = isBatch
                    ? succeeded.map(item => item?.result || {}).filter(result => result.partial_success || result.repair_required || statusNeedsRepair(result.projection_status))
                    : [data].filter(result => result.partial_success || result.repair_required || statusNeedsRepair(result.projection_status));
                const failedCount = Array.isArray(data.failed) ? data.failed.length : 0;
                if (partialResults.length) {
                    feishuMsg.value = `撤销已生效，派生同步待修复（${partialResults.length} 项）${failedCount ? `；另有 ${failedCount} 项撤销失败` : ''}`;
                } else if (failedCount) {
                    feishuMsg.value = `部分撤销完成：${succeeded.length} 项成功，${failedCount} 项失败`;
                } else {
                    feishuMsg.value = data.message || `已撤销 ${targets.length} 项知识资产`;
                }
                feishuUndoAsset.value = null;
                const targetIds = new Set(targets.map(asset => asset.asset_id));
                feishuSelectedAssetIds.value = feishuSelectedAssetIds.value.filter(id => !targetIds.has(id));
                feishuRevokeConfirmOpen.value = false;
                feishuRevokeReason.value = '';
                feishuRevokeTargets.value = [];
                await loadFeishuWorkspace();
                await loadDocuments();
                await loadStats();
                await loadGraph();
            } catch (error) {
                feishuMsg.value = '撤销失败：' + error.message;
            } finally {
                feishuBatchLoading.value = false;
            }
        }

        function toggleFeishuAsset(assetId) {
            feishuSelectedAssetIds.value = feishuSelectedAssetIds.value.includes(assetId)
                ? feishuSelectedAssetIds.value.filter((id) => id !== assetId)
                : [...feishuSelectedAssetIds.value, assetId];
        }

        function toggleAllFeishuAssets() {
            feishuSelectedAssetIds.value = feishuAllAssetsSelected.value
                ? []
                : feishuPublishedAssets.value.map((asset) => asset.asset_id);
        }

        function batchRevertFeishuAssets() {
            if (!feishuSelectedAssetIds.value.length) {
                feishuMsg.value = '请至少选择一个已入库知识资产';
                return;
            }
            const selected = feishuPublishedAssets.value.filter(asset => feishuSelectedAssetIds.value.includes(asset.asset_id));
            openFeishuRevokeConfirm(selected);
        }

        function feishuMessageTone(message) {
            const text = String(message || '');
            if (/失败|必须|无权|不存在/.test(text) && !/撤销已生效/.test(text)) return 'err';
            if (/待修复|部分|待处理/.test(text)) return 'warn';
            return 'ok';
        }

        async function syncFeishuMessages(chatId = feishuSelectedGroupId.value) {
            const targetChatId = typeof chatId === 'string' ? chatId : feishuSelectedGroupId.value;
            if (!targetChatId) {
                feishuMsg.value = '请先选择一个项目群';
                return;
            }
            feishuSyncing.value = true;
            feishuMsg.value = '';
            try {
                const form = new FormData();
                form.append('page_size', '50');
                const res = await apiFetch(`${API_BASE}/feishu/groups/${encodeURIComponent(targetChatId)}/sync`, {
                    method: 'POST', body: form,
                });
                const data = await res.json();
                if (!data.success) throw new Error(data.detail || '群消息同步失败');
                feishuMsg.value = `${data.message || '飞书历史同步已进入处理队列'}${data.job?.job_id ? ` · ${data.job.job_id}` : ''}`;
                await loadProcessingJobs({ keepSelection: false, silent: true });
            } catch (e) {
                feishuMsg.value = '同步失败：' + e.message;
            } finally {
                feishuSyncing.value = false;
            }
        }

        async function reviewFeishuMessage(message, action) {
            if (!message?.message_id) return;
            feishuReviewingId.value = message.message_id;
            try {
                const sourceGroup = feishuGroups.value.find((group) => group.chat_id === message.chat_id);
                const category = sourceGroup?.default_category || 'proj_communication';
                const res = await apiFetch(`${API_BASE}/feishu/messages/${encodeURIComponent(message.message_id)}/review`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ action, category }),
                });
                const data = await res.json();
                if (!data.success) throw new Error(data.detail || '消息处理失败');
                feishuMsg.value = data.message || '消息状态已更新';
                await loadFeishuWorkspace();
                if (action === 'approve') {
                    await loadDocuments();
                    await loadStats();
                    await loadGraph();
                }
            } catch (e) {
                feishuMsg.value = '处理失败：' + e.message;
            } finally {
                feishuReviewingId.value = '';
            }
        }

        function feishuResourceUrls(message) {
            const resources = Array.isArray(message?.resource_files) ? message.resource_files : [];
            if (!message?.message_id) return [];
            return resources.flatMap((item, index) => item?.stored_file ? [{
                key: item.resource_key || `${message.message_id}-${index}`,
                url: `${API_BASE}/feishu/messages/${encodeURIComponent(message.message_id)}/resources/${index}`,
            }] : []);
        }

        function feishuVisibleLinks(message) {
            const unique = new Map();
            for (const item of Array.isArray(message?.links) ? message.links : []) {
                try {
                    const url = new URL(String(item?.href || ''));
                    if (!['http:', 'https:'].includes(url.protocol)) continue;
                    unique.set(url.href, { href: url.href, text: String(item?.text || url.hostname) });
                } catch (_) {
                    // 忽略无效或不安全链接。
                }
            }
            return [...unique.values()];
        }

        function feishuDisplayContent(message) {
            const text = String(message?.content || '');
            const title = String(message?.title || '').trim();
            if (!title) return text;
            const heading = `# ${title}`;
            return text.startsWith(heading) ? text.slice(heading.length).trimStart() : text;
        }

        function feishuExtractionLabel(message) {
            const status = message?.extraction_status;
            const method = String(message?.extraction_method || '');
            if (status === 'queued') return '图片已归档，等待后台识别';
            if (status === 'failed' && Number(message?.resource_count || 0) > 0) return '原图已保存，内容识别待重试';
            if (status === 'failed') return '图片保存或识别失败';
            if (status === 'partial') return '部分图片待处理';
            if (status === 'legacy_pending') return '历史内容待重新同步';
            if (method.includes('local_rapidocr')) return '本地 OCR 已识别';
            if (method.includes('feishu_ocr')) return '飞书 OCR 已识别';
            if (method.includes('vision_model')) return '多模态模型已识别';
            if (method.includes('rich_text_parser')) return '富文本已解析';
            return '';
        }

        async function retryFeishuEnrichment(message) {
            if (!message?.message_id) return;
            feishuEnrichingId.value = message.message_id;
            feishuMsg.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/feishu/messages/${encodeURIComponent(message.message_id)}/enrichments`, { method: 'POST' });
                const data = await res.json();
                if (!res.ok || !data.success) throw new Error(data.detail || '重新识别提交失败');
                const acceptedJob = data.job;
                processingJobStatusFilter.value = 'active';
                processingJobTypeFilter.value = 'feishu_message_enrichment';
                processingJobKeyword.value = '';
                processingJobMessage.value = data.message || '图片识别已进入处理任务';
                tab.value = 'jobs';
                await loadProcessingJobs({ keepSelection: false });
                if (acceptedJob?.job_id) {
                    await openProcessingJob(acceptedJob, { silent: true });
                    trackFeishuEnrichmentJob(acceptedJob.job_id, message.message_id);
                }
            } catch (error) {
                feishuMsg.value = '重新识别提交失败：' + error.message;
                await loadFeishuWorkspace();
            } finally {
                feishuEnrichingId.value = '';
            }
        }

        async function trackFeishuEnrichmentJob(jobId, messageId) {
            let consecutiveErrors = 0;
            for (let attempt = 0; attempt < 300; attempt += 1) {
                await new Promise(resolve => setTimeout(resolve, 1000));
                try {
                    const res = await apiFetch(`${API_BASE}/processing/jobs/${encodeURIComponent(jobId)}`);
                    const data = await res.json().catch(() => ({}));
                    if (!res.ok || !data.success) throw new Error(data.detail || '任务状态读取失败');
                    consecutiveErrors = 0;
                    const job = data.job;
                    if (selectedProcessingJob.value?.job_id === jobId) selectedProcessingJob.value = job;
                    if (!['succeeded', 'failed', 'cancelled'].includes(job.status)) continue;
                    processingJobStatusFilter.value = '';
                    const result = job.result || {};
                    if (job.status === 'succeeded') {
                        processingJobMessage.value = result.message || '图片与富文本内容已完成后台识别';
                    } else if (job.status === 'cancelled') {
                        processingJobMessage.value = `消息 ${messageId} 的识别任务已取消`;
                    } else {
                        processingJobMessage.value = `图片识别失败：${job.last_error || '请检查 OCR 或多模态模型配置后重试'}`;
                    }
                    await loadFeishuWorkspace();
                    if (Number(result.published || 0) > 0) {
                        await Promise.all([loadDocuments(), loadStats(), loadGraph()]);
                    }
                    await loadProcessingJobs({ silent: true });
                    return;
                } catch (error) {
                    consecutiveErrors += 1;
                    if (consecutiveErrors >= 5) {
                        processingJobMessage.value = '图片识别任务状态暂时无法读取，可稍后刷新任务中心';
                        return;
                    }
                }
            }
            processingJobMessage.value = '图片识别仍在处理中，可稍后在任务中心继续查看';
        }

        function feishuModeLabel(mode) {
            return ({ off: '关闭采集', review: '谨慎沉淀', auto: '智能沉淀', archive_only: '仅归档' })[mode] || '智能沉淀';
        }

        function feishuReviewLabel(status) {
            return ({ pending: '待审核', approved: '已入库', ignored: '已忽略' })[status] || '待处理';
        }

        function feishuMessageTypeLabel(type) {
            return ({ text: '文本', image: '图片', file: '文件', post: '富文本', interactive: '卡片', share_chat: '聊天记录' })[type] || type || '消息';
        }

        function feishuContentStatusLabel(status) {
            return ({
                archived: '仅归档', excluded: '已自动排除', candidate: '聚合中', review_required: '需处理',
                published: '已入库', failed: '处理失败', reverted: '已撤销', not_collected: '未采集',
            })[status] || '已归档';
        }

        function feishuCandidateStatusLabel(status) {
            return ({
                ready_for_auto: '等待自动入库', review_required: '需处理', published: '已入库',
                failed: '处理失败', excluded: '已排除', reverted: '已撤销',
            })[status] || '需处理';
        }

        function feishuAuditLabel(action) {
            return ({
                item_received: '接收内容', candidate_created: '创建知识候选', candidate_updated: '更新知识候选',
                asset_published: '知识入库', asset_reverted: '撤销知识', candidate_exclude: '排除候选',
                candidate_category: '修改分类', candidate_retry: '重试候选', source_policy_saved: '更新采集规则',
                permissions_checked: '检测权限', replied: '机器人已回复', reply_failed: '机器人回复失败',
                item_enriched: '重新识别内容', image_collected: '图片已接收',
            })[action] || feishuActionLabel(action);
        }

        function formatFeishuTime(value) {
            const raw = String(value || '').trim();
            if (/^\d{10,}$/.test(raw)) {
                const number = Number(raw);
                const date = new Date(number > 9999999999 ? number : number * 1000);
                if (!Number.isNaN(date.getTime())) {
                    return date.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false });
                }
            }
            return raw || '刚刚';
        }

        // ── 离职交接 ──

        function onResignFilesChange(e) {
            const files = e.target.files;
            if (files && files.length > 0) {
                resignFiles.value = Array.from(files);
            }
        }

        function processingJobStatusLabel(status) {
            if (status === 'interrupted') return '执行中断';
            return processingJobStatusMeta[status]?.label || status || '未知状态';
        }

        function processingJobTypeLabel(type) {
            return processingJobTypeMeta[type] || type || '后台处理';
        }

        function processingJobDisplayName(job) {
            if (!job) return '处理任务';
            return job.display_name
                || job.payload?.original_filename
                || job.result?.filename
                || job.payload?.filename
                || job.payload?.display_name
                || job.payload?.title
                || job.source_id
                || job.linked_asset_id
                || processingJobTypeLabel(job.job_type);
        }

        function processingJobStageLabel(stage) {
            return processingJobStageMeta[stage] || stage || '等待处理';
        }

        async function loadProcessingJobs(options = {}) {
            if (!can('job.read')) return;
            if (!options.silent) processingJobsLoading.value = true;
            processingJobsError.value = '';
            const params = new URLSearchParams({
                status: processingJobStatusFilter.value,
                job_type: processingJobTypeFilter.value,
                keyword: processingJobKeyword.value.trim(),
                page: String(processingJobPage.value),
                page_size: '40',
            });
            try {
                const res = await apiFetch(`${API_BASE}/processing/jobs?${params}`);
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '处理任务加载失败');
                processingJobs.value = data.items || [];
                processingJobSummary.value = data.summary || {};
                processingJobPage.value = data.page || 1;
                processingJobPages.value = data.pages || 1;
                processingJobTotal.value = data.total || 0;
                if (selectedProcessingJob.value && options.keepSelection !== false) {
                    const current = processingJobs.value.find(item => item.job_id === selectedProcessingJob.value.job_id);
                    if (current) await openProcessingJob(current, { silent: true });
                    else selectedProcessingJob.value = null;
                }
            } catch (error) {
                processingJobsError.value = error.message || '处理任务加载失败';
            } finally {
                processingJobsLoading.value = false;
            }
        }

        async function openProcessingJob(job, options = {}) {
            if (!job?.job_id) return;
            if (!options.silent) processingJobDetailLoading.value = true;
            processingJobMessage.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/processing/jobs/${encodeURIComponent(job.job_id)}`);
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '任务详情加载失败');
                selectedProcessingJob.value = data.job;
            } catch (error) {
                processingJobMessage.value = error.message || '任务详情加载失败';
            } finally {
                processingJobDetailLoading.value = false;
            }
        }

        async function actOnProcessingJob(action) {
            const job = selectedProcessingJob.value;
            if (!job?.job_id || processingJobActionLoading.value) return;
            processingJobActionLoading.value = action;
            processingJobMessage.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/processing/jobs/${encodeURIComponent(job.job_id)}/${action}`, { method: 'POST' });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '任务操作失败');
                processingJobMessage.value = data.message || '任务状态已更新';
                selectedProcessingJob.value = data.job;
                await loadProcessingJobs();
            } catch (error) {
                processingJobMessage.value = error.message || '任务操作失败';
            } finally {
                processingJobActionLoading.value = '';
            }
        }

        async function changeProcessingJobPage(delta) {
            const nextPage = Math.max(1, Math.min(processingJobPages.value, processingJobPage.value + delta));
            if (nextPage === processingJobPage.value) return;
            processingJobPage.value = nextPage;
            selectedProcessingJob.value = null;
            await loadProcessingJobs({ keepSelection: false });
        }

        async function loadHandoverRecords() {
            handoverLoading.value = true;
            handoverLoadError.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/resignation/records`);
                const data = await res.json();
                if (!res.ok || !data.success) throw new Error(data.detail || '交接记录加载失败');
                handoverRecords.value = data.records || [];
                applyHandoverStage(handoverRecords.value[0]);
            } catch (e) {
                handoverRecords.value = [];
                handoverLoadError.value = '交接记录暂时无法读取：' + e.message;
            } finally {
                handoverLoading.value = false;
            }
        }

        function replaceHandoverRecord(record) {
            if (!record?.id) return;
            handoverRecords.value = handoverRecords.value.map(item => item.id === record.id ? record : item);
            if (selectedHandover.value?.id === record.id) selectedHandover.value = record;
            resignResult.value = resignResult.value?.handover?.id === record.id
                ? { ...resignResult.value, handover: record }
                : resignResult.value;
            applyHandoverStage(record);
        }

        function initializeHandoverDrafts(record) {
            handoverItemDrafts.value = Object.fromEntries(
                (record?.items || []).map(item => [
                    item.item_id,
                    item.submission_draft?.evidence ?? item.evidence ?? '',
                ]),
            );
            handoverItemAssetRefs.value = Object.fromEntries(
                (record?.items || []).map(item => [
                    item.item_id, Array.isArray(item.submission_draft?.asset_refs)
                        ? item.submission_draft.asset_refs : [],
                ]),
            );
            handoverItemAssetPickers.value = {};
            handoverItemDraftSaving.value = {};
            handoverItemDraftMessages.value = {};
            handoverRejectDrafts.value = Object.fromEntries(
                (record?.items || []).map(item => [item.item_id, '']),
            );
            handoverProxyReasonDrafts.value = Object.fromEntries(
                (record?.items || []).map(item => [item.item_id, '']),
            );
            handoverInventoryExcludeDrafts.value = Object.fromEntries(
                (record?.inventory || []).map(item => [item.inventory_id, item.exclusion_reason || '']),
            );
            handoverInventoryExcludeActive.value = {};
            handoverRiskActionDrafts.value = Object.fromEntries(
                (record?.risk_items || []).map(risk => [risk.risk_id, {
                    mitigation: risk.mitigation || '', close_evidence: risk.close_evidence || '', reason: '',
                }]),
            );
            handoverRiskDraft.value = {
                title: '', impact: '', severity: 'medium',
                owner_user_id: currentUser.value?.user_id || '', due_at: record?.due_date || '', mitigation: '',
            };
        }

        function closeHandoverDetail() {
            showHandoverModal.value = false;
            selectedHandover.value = null;
            handoverActionMessage.value = '';
            handoverDetailLoading.value = false;
        }

        async function openHandoverDetail(record) {
            if (!record?.id) return;
            showHandoverModal.value = true;
            selectedHandover.value = record;
            handoverDetailLoading.value = true;
            handoverActionMessage.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/resignation/${encodeURIComponent(record.id)}`);
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '交接详情加载失败');
                selectedHandover.value = data.record;
                replaceHandoverRecord(data.record);
                initializeHandoverDrafts(data.record);
                applyHandoverViewDefaults();
            } catch (error) {
                handoverActionMessage.value = error.message || '交接详情加载失败';
            } finally {
                handoverDetailLoading.value = false;
            }
        }

        async function actOnHandoverItem(item, action) {
            const record = selectedHandover.value;
            if (!record?.id || !item?.item_id || handoverActionLoading.value) return;
            const actionKey = `item:${item.item_id}:${action}`;
            handoverActionLoading.value = actionKey;
            handoverActionMessage.value = '';
            const payload = { action };
            if (action === 'submit') {
                payload.evidence = String(handoverItemDrafts.value[item.item_id] || '').trim();
                payload.asset_refs = (handoverItemAssetRefs.value[item.item_id] || []).map(reference => ({
                    asset_id: reference.asset_id, version_id: reference.version_id,
                    source_kind: reference.source_kind || 'existing',
                }));
            }
            if (action === 'reject') payload.reason = String(handoverRejectDrafts.value[item.item_id] || '').trim();
            if (['accept', 'reject'].includes(action) && item.review_action_mode === 'proxy') {
                payload.proxy_reason = String(handoverProxyReasonDrafts.value[item.item_id] || '').trim();
            }
            try {
                const res = await apiFetch(
                    `${API_BASE}/resignation/${encodeURIComponent(record.id)}/items/${encodeURIComponent(item.item_id)}/actions`,
                    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) },
                );
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || data.message || '交接项操作失败');
                replaceHandoverRecord(data.record);
                initializeHandoverDrafts(data.record);
                handoverActionMessage.value = data.message || '交接项已更新';
                if (data.task) await loadKnowledgeTasks({ keepSelection: false });
            } catch (error) {
                handoverActionMessage.value = error.message || '交接项操作失败';
            } finally {
                handoverActionLoading.value = '';
            }
        }

        function handoverItemHasDraftEvidence(item) {
            return Boolean(
                String(handoverItemDrafts.value[item?.item_id] || '').trim()
                || (handoverItemAssetRefs.value[item?.item_id] || []).length,
            );
        }

        function handoverEvidenceReferenceLabel(reference) {
            if (!reference) return '未命名资料';
            return reference.title || reference.source_file || '资料已不可访问';
        }

        function syncHandoverItemDraft(itemId, draft) {
            if (!itemId || !draft) return;
            handoverItemDrafts.value = {
                ...handoverItemDrafts.value,
                [itemId]: String(draft.evidence || ''),
            };
            handoverItemAssetRefs.value = {
                ...handoverItemAssetRefs.value,
                [itemId]: Array.isArray(draft.asset_refs) ? draft.asset_refs : [],
            };
        }

        async function loadHandoverItemSubmissionContext(item, { force = false } = {}) {
            const record = selectedHandover.value;
            const itemId = item?.item_id;
            if (!record?.id || !itemId) return;
            if (!force && Array.isArray(handoverItemAssetOptions.value[itemId])) return;
            handoverItemDraftSaving.value = { ...handoverItemDraftSaving.value, [itemId]: true };
            handoverItemDraftMessages.value = { ...handoverItemDraftMessages.value, [itemId]: '' };
            try {
                const res = await apiFetch(
                    `${API_BASE}/resignation/${encodeURIComponent(record.id)}/items/${encodeURIComponent(itemId)}/submission-context`,
                );
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '可关联资料加载失败');
                handoverItemAssetOptions.value = {
                    ...handoverItemAssetOptions.value,
                    [itemId]: Array.isArray(data.assets) ? data.assets : [],
                };
                syncHandoverItemDraft(itemId, data.draft);
                handoverItemDraftMessages.value = {
                    ...handoverItemDraftMessages.value,
                    [itemId]: data.message || '',
                };
            } catch (error) {
                handoverItemDraftMessages.value = {
                    ...handoverItemDraftMessages.value,
                    [itemId]: `无法加载可关联资料：${error.message || '未知错误'}`,
                };
            } finally {
                handoverItemDraftSaving.value = { ...handoverItemDraftSaving.value, [itemId]: false };
            }
        }

        async function saveHandoverItemDraft(item, { silent = false } = {}) {
            const record = selectedHandover.value;
            const itemId = item?.item_id;
            if (!record?.id || !itemId || handoverItemDraftSaving.value[itemId]) return;
            handoverItemDraftSaving.value = { ...handoverItemDraftSaving.value, [itemId]: true };
            try {
                const res = await apiFetch(
                    `${API_BASE}/resignation/${encodeURIComponent(record.id)}/items/${encodeURIComponent(itemId)}/draft`,
                    {
                        method: 'PUT', headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            evidence: String(handoverItemDrafts.value[itemId] || '').trim(),
                            asset_refs: (handoverItemAssetRefs.value[itemId] || []).map(reference => ({
                                asset_id: reference.asset_id, version_id: reference.version_id,
                                source_kind: reference.source_kind || 'existing',
                            })),
                        }),
                    },
                );
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '提交草稿保存失败');
                syncHandoverItemDraft(itemId, data.draft);
                handoverItemDraftMessages.value = {
                    ...handoverItemDraftMessages.value,
                    [itemId]: silent ? '草稿已自动保存' : (data.message || '提交草稿已保存'),
                };
            } catch (error) {
                handoverItemDraftMessages.value = {
                    ...handoverItemDraftMessages.value,
                    [itemId]: `草稿未保存：${error.message || '未知错误'}`,
                };
            } finally {
                handoverItemDraftSaving.value = { ...handoverItemDraftSaving.value, [itemId]: false };
            }
        }

        function scheduleHandoverItemDraftSave(item) {
            const itemId = item?.item_id;
            if (!itemId) return;
            const previous = handoverItemDraftTimers.get(itemId);
            if (previous) clearTimeout(previous);
            handoverItemDraftTimers.set(itemId, setTimeout(() => {
                handoverItemDraftTimers.delete(itemId);
                saveHandoverItemDraft(item, { silent: true });
            }, 550));
        }

        function addHandoverItemExistingAsset(item) {
            const itemId = item?.item_id;
            const assetId = String(handoverItemAssetPickers.value[itemId] || '');
            const option = (handoverItemAssetOptions.value[itemId] || []).find(
                candidate => candidate.asset_id === assetId,
            );
            if (!option) return;
            const current = handoverItemAssetRefs.value[itemId] || [];
            if (current.some(reference => reference.asset_id === option.asset_id && reference.version_id === option.version_id)) {
                handoverItemDraftMessages.value = { ...handoverItemDraftMessages.value, [itemId]: '该资料已关联到当前草稿' };
                return;
            }
            if (current.length >= 10) {
                handoverItemDraftMessages.value = { ...handoverItemDraftMessages.value, [itemId]: '每个必交项最多关联 10 份资料' };
                return;
            }
            handoverItemAssetRefs.value = {
                ...handoverItemAssetRefs.value,
                [itemId]: [...current, { ...option, source_kind: 'existing' }],
            };
            handoverItemAssetPickers.value = { ...handoverItemAssetPickers.value, [itemId]: '' };
            saveHandoverItemDraft(item);
        }

        function removeHandoverItemDraftAsset(item, reference) {
            const itemId = item?.item_id;
            if (!itemId) return;
            handoverItemAssetRefs.value = {
                ...handoverItemAssetRefs.value,
                [itemId]: (handoverItemAssetRefs.value[itemId] || []).filter(candidate => !(
                    candidate.asset_id === reference.asset_id && candidate.version_id === reference.version_id
                )),
            };
            saveHandoverItemDraft(item);
        }

        async function uploadHandoverItemEvidenceFiles(item, event) {
            const record = selectedHandover.value;
            const itemId = item?.item_id;
            const files = Array.from(event?.target?.files || []);
            if (!record?.id || !itemId || !files.length || handoverItemDraftSaving.value[itemId]) return;
            const attachedCount = (handoverItemAssetRefs.value[itemId] || []).length;
            if (attachedCount + files.length > 10) {
                handoverItemDraftMessages.value = {
                    ...handoverItemDraftMessages.value,
                    [itemId]: `每个必交项最多关联 10 份资料；当前已有 ${attachedCount} 份，请减少本次上传数量或先移除资料。`,
                };
                if (event?.target) event.target.value = '';
                return;
            }
            handoverItemDraftSaving.value = { ...handoverItemDraftSaving.value, [itemId]: true };
            handoverItemDraftMessages.value = { ...handoverItemDraftMessages.value, [itemId]: '' };
            try {
                for (const file of files) {
                    const form = new FormData();
                    form.append('file', file);
                    const res = await apiFetch(
                        `${API_BASE}/resignation/${encodeURIComponent(record.id)}/items/${encodeURIComponent(itemId)}/attachments`,
                        { method: 'POST', body: form },
                    );
                    const data = await res.json().catch(() => ({}));
                    if (!res.ok || !data.success) throw new Error(data.detail || `${file.name} 上传失败`);
                    syncHandoverItemDraft(itemId, data.draft);
                }
                handoverItemDraftMessages.value = {
                    ...handoverItemDraftMessages.value,
                    [itemId]: `已上传并关联 ${files.length} 个补充文件`,
                };
                await Promise.all([loadDocuments(), loadGroupedDocs(), loadKnowledgeAssets(), loadStats()]);
                await loadHandoverItemSubmissionContext(item, { force: true });
            } catch (error) {
                handoverItemDraftMessages.value = {
                    ...handoverItemDraftMessages.value,
                    [itemId]: `上传未完成：${error.message || '未知错误'}`,
                };
            } finally {
                handoverItemDraftSaving.value = { ...handoverItemDraftSaving.value, [itemId]: false };
                if (event?.target) event.target.value = '';
            }
        }

        function handoverInventorySourceLabel(sourceType) {
            return ({ knowledge_asset: '知识资产', knowledge_task: '未完成任务', graph_topic: '图谱主题' })[sourceType] || sourceType;
        }

        function handoverInventoryStatusLabel(status) {
            return ({ pending: '待审阅', included: '已纳入', excluded: '已排除' })[status] || status;
        }

        async function refreshHandoverInventory() {
            const record = selectedHandover.value;
            if (!record?.id || handoverActionLoading.value) return;
            handoverActionLoading.value = 'inventory:refresh';
            handoverActionMessage.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/resignation/${encodeURIComponent(record.id)}/inventory/refresh`, { method: 'POST' });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || data.message || '知识盘点刷新失败');
                replaceHandoverRecord(data.record);
                initializeHandoverDrafts(data.record);
                handoverActionMessage.value = data.message || '知识盘点已刷新';
            } catch (error) {
                handoverActionMessage.value = error.message || '知识盘点刷新失败';
            } finally {
                handoverActionLoading.value = '';
            }
        }

        async function actOnHandoverInventory(entry, action) {
            const record = selectedHandover.value;
            if (!record?.id || !entry?.inventory_id || handoverActionLoading.value) return;
            handoverActionLoading.value = `inventory:${entry.inventory_id}:${action}`;
            handoverActionMessage.value = '';
            const payload = { action };
            if (action === 'exclude') payload.reason = String(handoverInventoryExcludeDrafts.value[entry.inventory_id] || '').trim();
            try {
                const res = await apiFetch(
                    `${API_BASE}/resignation/${encodeURIComponent(record.id)}/inventory/${encodeURIComponent(entry.inventory_id)}/actions`,
                    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) },
                );
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || data.message || '知识盘点操作失败');
                replaceHandoverRecord(data.record);
                initializeHandoverDrafts(data.record);
                handoverActionMessage.value = data.message || '知识盘点已更新';
            } catch (error) {
                handoverActionMessage.value = error.message || '知识盘点操作失败';
            } finally {
                handoverActionLoading.value = '';
            }
        }

        async function createHandoverRisk() {
            const record = selectedHandover.value;
            if (!record?.id || handoverActionLoading.value) return;
            handoverActionLoading.value = 'risk:create';
            handoverActionMessage.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/resignation/${encodeURIComponent(record.id)}/risks`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(handoverRiskDraft.value),
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || data.message || '风险登记失败');
                replaceHandoverRecord(data.record);
                initializeHandoverDrafts(data.record);
                handoverActionMessage.value = data.message || '风险已登记';
            } catch (error) {
                handoverActionMessage.value = error.message || '风险登记失败';
            } finally {
                handoverActionLoading.value = '';
            }
        }

        async function actOnHandoverRisk(risk, action) {
            const record = selectedHandover.value;
            if (!record?.id || !risk?.risk_id || handoverActionLoading.value) return;
            const actionKey = `risk:${risk.risk_id}:${action}`;
            handoverActionLoading.value = actionKey;
            handoverActionMessage.value = '';
            const draft = handoverRiskActionDrafts.value[risk.risk_id] || {};
            const payload = { action, ...draft };
            try {
                const res = await apiFetch(
                    `${API_BASE}/resignation/${encodeURIComponent(record.id)}/risks/${encodeURIComponent(risk.risk_id)}/actions`,
                    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) },
                );
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || data.message || '风险操作失败');
                replaceHandoverRecord(data.record);
                initializeHandoverDrafts(data.record);
                handoverActionMessage.value = data.message || '风险已更新';
            } catch (error) {
                handoverActionMessage.value = error.message || '风险操作失败';
            } finally {
                handoverActionLoading.value = '';
            }
        }

        async function completeHandover() {
            const record = selectedHandover.value;
            if (!record?.id || handoverActionLoading.value) return;
            if (!window.confirm('关闭后交接项、风险和资产版本将封存为不可变快照。确认继续？')) return;
            handoverActionLoading.value = 'complete';
            handoverActionMessage.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/resignation/${encodeURIComponent(record.id)}/complete`, {
                    method: 'POST',
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || data.message || '交接关闭失败');
                replaceHandoverRecord(data.record);
                handoverActionMessage.value = data.message || '交接已关闭并封存';
            } catch (error) {
                handoverActionMessage.value = error.message || '交接关闭失败';
            } finally {
                handoverActionLoading.value = '';
            }
        }

        async function acceptHandover(record) {
            if (!record?.id || acceptingHandoverId.value) return;
            acceptingHandoverId.value = record.id;
            handoverActionMessage.value = '';
            resignError.value = '';
            try {
                const form = new FormData();
                form.append('accepted_by', record.recipient || '接替人');
                const res = await apiFetch(`${API_BASE}/resignation/${encodeURIComponent(record.id)}/accept`, { method: 'POST', body: form });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || data.message || '确认失败');
                replaceHandoverRecord(data.record);
                if (selectedHandover.value?.id === record.id) initializeHandoverDrafts(data.record);
                handoverActionMessage.value = data.message || '交接状态已更新';
            } catch (e) {
                resignError.value = '确认接收失败: ' + e.message;
            } finally {
                acceptingHandoverId.value = '';
            }
        }

        async function cancelHandover(record) {
            if (!canCancelHandover(record) || cancellingHandoverId.value) return;
            if (!window.confirm('确定取消这项离职交接吗？已上传的知识文档会继续保留在知识库中。')) return;
            cancellingHandoverId.value = record.id;
            handoverActionMessage.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/resignation/${record.id}/cancel`, { method: 'POST' });
                const data = await res.json().catch(() => ({}));
                if (!res.ok) throw new Error(data.detail || '取消交接失败');
                if (selectedHandover.value?.id === record.id) closeHandoverDetail();
                await loadHandoverRecords();
                handoverActionMessage.value = data.message || '交接已取消';
            } catch (error) {
                handoverActionMessage.value = error.message || '取消交接失败';
            } finally {
                cancellingHandoverId.value = '';
            }
        }

        async function openHandoverLearningPlan(record) {
            if (!record?.recipient_user_id || !record?.role_key) return;
            onboardingTargetUserId.value = record.recipient_user_id;
            onboardingRole.value = record.role_key;
            navigateTo('onboarding');
            await loadOnboardingGuide({ openPlanDetail: true });
        }

        async function loadOnboardingTemplates() {
            try {
                const res = await apiFetch(`${API_BASE}/onboarding/templates`);
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '岗位模板加载失败');
                onboardingTemplates.value = data.templates || [];
                if (!onboardingTemplates.value.some(item => item.role_key === onboardingRole.value)) {
                    onboardingRole.value = onboardingTemplates.value[0]?.role_key || 'developer';
                }
                if (!onboardingTargetUserId.value) onboardingTargetUserId.value = currentUser.value?.user_id || '';
            } catch (error) {
                onboardingError.value = `加载失败: ${error.message || '岗位模板加载失败'}`;
            }
        }

        async function loadOnboardingGuide({ openPlanDetail = false } = {}) {
            onboardingLoading.value = true;
            onboardingError.value = '';
            onboardingTaskMessage.value = '';
            try {
                if (!onboardingTemplates.value.length) await loadOnboardingTemplates();
                const params = new URLSearchParams({ role: onboardingRole.value });
                if (onboardingTargetUserId.value) params.set('user_id', onboardingTargetUserId.value);
                const res = await apiFetch(`${API_BASE}/onboarding/guide?${params}`);
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '学习路径加载失败');
                onboardingData.value = data;
                onboardingEvidenceDrafts.value = Object.fromEntries(
                    (data.plan?.items || []).map(item => [item.item_id, item.evidence || '']),
                );
                if (openPlanDetail && can('onboarding.manage')) {
                    onboardingPlanDetailId.value = data.plan?.plan_id || 'new';
                }
            } catch (e) {
                onboardingError.value = '加载失败: ' + e.message;
            } finally {
                onboardingLoading.value = false;
            }
        }

        async function loadSupervisedOnboardingPlans() {
            if (!can('onboarding.manage')) {
                supervisedOnboardingPlans.value = [];
                return;
            }
            supervisedOnboardingPlansLoading.value = true;
            supervisedOnboardingPlansError.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/onboarding/supervised-plans`);
                const data = await res.json().catch(() => ({}));
                if (res.status === 404) {
                    throw new Error('主管学习计划服务尚未更新，请重启项目服务后重新打开页面');
                }
                if (!res.ok || !data.success) throw new Error(data.detail || '主管学习计划加载失败');
                supervisedOnboardingPlans.value = data.plans || [];
            } catch (error) {
                supervisedOnboardingPlansError.value = error.message || '主管学习计划加载失败';
            } finally {
                supervisedOnboardingPlansLoading.value = false;
            }
        }

        async function openOnboardingPlanAssignment() {
            const preferredMember = onboardingTargetMembers.value.find(member => (
                ['new_member', 'project_member', 'handover_participant'].includes(member.role)
            ));
            if (!onboardingTargetUserId.value || onboardingTargetUserId.value === currentUser.value?.user_id) {
                onboardingTargetUserId.value = preferredMember?.user_id || onboardingTargetMembers.value[0]?.user_id || '';
            }
            onboardingPlanDetailId.value = 'new';
            await loadOnboardingGuide({ openPlanDetail: true });
        }

        async function openSupervisedOnboardingPlan(plan) {
            if (!plan?.user_id) return;
            onboardingTargetUserId.value = plan.user_id;
            onboardingRole.value = plan.role_key || onboardingRole.value;
            onboardingPlanDetailId.value = plan.plan_id || 'new';
            await loadOnboardingGuide({ openPlanDetail: true });
        }

        function closeOnboardingPlanDetail() {
            onboardingPlanDetailId.value = '';
            onboardingTaskMessage.value = '';
        }

        async function createOnboardingPlan() {
            if (!onboardingData.value?.template?.template_id || !onboardingTargetUserId.value) return;
            onboardingActionLoading.value = 'create-plan';
            onboardingTaskMessage.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/onboarding/plans`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        user_id: onboardingTargetUserId.value,
                        template_id: onboardingData.value.template.template_id,
                    }),
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '学习计划创建失败');
                onboardingData.value = { ...onboardingData.value, plan: data.plan };
                onboardingEvidenceDrafts.value = Object.fromEntries(
                    (data.plan?.items || []).map(item => [item.item_id, item.evidence || '']),
                );
                onboardingPlanDetailId.value = data.plan?.plan_id || onboardingPlanDetailId.value;
                await loadSupervisedOnboardingPlans();
                onboardingTaskMessage.value = data.message || '学习计划已创建';
            } catch (error) {
                onboardingTaskMessage.value = `操作失败：${error.message || '学习计划创建失败'}`;
            } finally {
                onboardingActionLoading.value = '';
            }
        }

        async function updateOnboardingItem(item, status) {
            const plan = onboardingData.value?.plan;
            if (!plan?.plan_id || !item?.item_id || onboardingActionLoading.value) return;
            onboardingActionLoading.value = item.item_id;
            onboardingTaskMessage.value = '';
            try {
                const res = await apiFetch(
                    `${API_BASE}/onboarding/plans/${encodeURIComponent(plan.plan_id)}/items/${encodeURIComponent(item.item_id)}`,
                    {
                        method: 'PATCH', headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            status,
                            evidence: onboardingEvidenceDrafts.value[item.item_id] || item.evidence || '',
                        }),
                    },
                );
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '学习进度更新失败');
                onboardingData.value = { ...onboardingData.value, plan: data.plan };
                await loadSupervisedOnboardingPlans();
                onboardingTaskMessage.value = data.message || '学习进度已更新';
            } catch (error) {
                onboardingTaskMessage.value = `操作失败：${error.message || '学习进度更新失败'}`;
            } finally {
                onboardingActionLoading.value = '';
            }
        }

        async function refreshOnboardingPlanMaterials() {
            const plan = onboardingData.value?.plan;
            if (!plan?.plan_id || onboardingActionLoading.value) return;
            onboardingActionLoading.value = 'refresh-materials';
            onboardingTaskMessage.value = '';
            try {
                const res = await apiFetch(
                    `${API_BASE}/onboarding/plans/${encodeURIComponent(plan.plan_id)}/refresh-materials`,
                    { method: 'POST' },
                );
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '学习资料刷新失败');
                onboardingData.value = { ...onboardingData.value, plan: data.plan };
                onboardingEvidenceDrafts.value = Object.fromEntries(
                    (data.plan?.items || []).map(item => [item.item_id, item.evidence || '']),
                );
                await loadSupervisedOnboardingPlans();
                onboardingTaskMessage.value = data.message || '学习资料已刷新';
            } catch (error) {
                onboardingTaskMessage.value = `操作失败：${error.message || '学习资料刷新失败'}`;
            } finally {
                onboardingActionLoading.value = '';
            }
        }

        async function confirmSupervisedOnboardingPlan(plan) {
            const confirmation = plan?.manager_confirmation || {};
            if (!plan?.plan_id || !confirmation.item_id || !confirmation.can_confirm || onboardingActionLoading.value) return;
            const confirmed = window.confirm(`确认 ${plan.user_name || '该成员'} 已完成「${plan.role_name || '岗位'}」学习计划，并达到岗位上手标准吗？`);
            if (!confirmed) return;
            onboardingActionLoading.value = `manager-confirmation:${plan.plan_id}`;
            onboardingTaskMessage.value = '';
            try {
                const res = await apiFetch(
                    `${API_BASE}/onboarding/plans/${encodeURIComponent(plan.plan_id)}/items/${encodeURIComponent(confirmation.item_id)}`,
                    {
                        method: 'PATCH', headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            status: 'completed',
                            evidence: '负责人已在主管学习计划列表中确认：阅读与实践任务均已完成。',
                        }),
                    },
                );
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '负责人确认失败');
                if (onboardingData.value?.plan?.plan_id === plan.plan_id) {
                    onboardingData.value = { ...onboardingData.value, plan: data.plan };
                    onboardingPlanDetailId.value = '';
                }
                await loadSupervisedOnboardingPlans();
                onboardingTaskMessage.value = `${plan.user_name || '成员'}的学习计划已负责人确认`;
            } catch (error) {
                onboardingTaskMessage.value = `操作失败：${error.message || '负责人确认失败'}`;
            } finally {
                onboardingActionLoading.value = '';
            }
        }

        function onboardingItemTypeLabel(type) {
            return { reading: '必读资料', practice: '实践任务', manager_confirmation: '负责人确认' }[type] || '学习任务';
        }

        function readinessDimensionLabel(key) {
            return {
                topic_coverage: '主题覆盖', authority: '权威性',
                freshness: '新鲜度', traceability: '可追溯性',
            }[key] || key;
        }

        function onboardingItemStatusLabel(status) {
            return { pending: '待完成', completed: '已完成', blocked: '待补充资料' }[status] || '待处理';
        }

        function openOnboardingTemplateEditor() {
            const template = onboardingData.value?.template;
            if (!template) return;
            onboardingTemplateDraft.value = JSON.parse(JSON.stringify(template));
            showOnboardingTemplateModal.value = true;
            onboardingTaskMessage.value = '';
        }

        function addOnboardingTopic() {
            if (!onboardingTemplateDraft.value) return;
            onboardingTemplateDraft.value.topics.push({
                topic_key: `topic_${Date.now()}`, label: '', weight: 1,
                required_asset_ids: [], practice_task: '', owner_user_id: '', completion_standard: '',
            });
        }

        function removeOnboardingTopic(index) {
            if ((onboardingTemplateDraft.value?.topics || []).length <= 1) {
                onboardingTaskMessage.value = '岗位模板至少保留一个知识主题';
                return;
            }
            onboardingTemplateDraft.value.topics.splice(index, 1);
        }

        async function saveOnboardingTemplate() {
            const draft = onboardingTemplateDraft.value;
            if (!draft?.template_id || onboardingActionLoading.value) return;
            onboardingActionLoading.value = 'save-template';
            onboardingTaskMessage.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/onboarding/templates/${encodeURIComponent(draft.template_id)}`, {
                    method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(draft),
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '岗位模板保存失败');
                showOnboardingTemplateModal.value = false;
                onboardingTaskMessage.value = data.message || '岗位模板已保存';
                await loadOnboardingTemplates();
                await loadOnboardingGuide();
            } catch (error) {
                onboardingTaskMessage.value = `操作失败：${error.message || '岗位模板保存失败'}`;
            } finally {
                onboardingActionLoading.value = '';
            }
        }

        function knowledgeTaskStatusLabel(status) {
            return knowledgeTaskStatusMeta[status]?.label || status || '未知';
        }

        function knowledgeTaskTypeLabel(type) {
            return knowledgeTaskTypeMeta[type] || type || '人工任务';
        }

        function knowledgeTaskPriorityLabel(priority) {
            return knowledgeTaskPriorityMeta[priority] || priority || '中';
        }

        function knowledgeTaskEventLabel(eventType) {
            return knowledgeTaskEventMeta[eventType] || eventType || '状态更新';
        }

        function knowledgeTaskNotificationKindLabel(kind) {
            return { manual: '手动提醒', due_soon: '到期提醒', overdue: '逾期提醒', escalation: '逾期升级' }[kind] || '任务提醒';
        }

        function knowledgeTaskNotificationStatusLabel(notification) {
            if (notification?.processing_status === 'retry_wait') return '等待自动重试';
            if (notification?.processing_status === 'running') return '正在投递';
            if (notification?.processing_status === 'queued') return '等待投递';
            const status = notification?.status;
            return { pending: '等待投递', sent: '已发送', failed: '最终失败', skipped: '已跳过' }[status] || status || '未知';
        }

        function knowledgeTaskWritebackTitle(writeback) {
            if (writeback?.type === 'asset_review_confirm') return '资产复审联动';
            if (writeback?.type === 'asset_version_publish') return '资产发布联动';
            return '交接联动';
        }

        function knowledgeTaskWritebackStatus(writeback) {
            if (!writeback?.applied) return '未自动更新';
            if (!['asset_review_confirm', 'asset_version_publish'].includes(writeback.type)) return '已重新提交';
            const status = writeback.processing_job?.status;
            const published = writeback.type === 'asset_version_publish' ? '已发布 · ' : '';
            if (status === 'succeeded') return `${published}${writeback.type === 'asset_review_confirm' ? '已刷新就绪度' : '投影已刷新'}`;
            if (status === 'failed') return `${published}刷新失败`;
            if (status === 'retry_wait') return `${published}等待重试`;
            if (status === 'running') return `${published}正在刷新`;
            return `${published}刷新已排队`;
        }

        async function openKnowledgeTaskProcessingJob(writeback) {
            const jobId = String(writeback?.processing_job_id || '');
            if (!jobId || !can('job.read')) return;
            processingJobStatusFilter.value = '';
            processingJobTypeFilter.value = 'projection_repair';
            processingJobKeyword.value = jobId;
            processingJobPage.value = 1;
            tab.value = 'jobs';
            await loadProcessingJobs({ keepSelection: false });
            const job = processingJobs.value.find(item => item.job_id === jobId);
            if (job) await openProcessingJob(job);
        }

        async function loadKnowledgeTaskNotificationPolicy() {
            knowledgeTaskPolicyLoading.value = true;
            try {
                const res = await apiFetch(`${API_BASE}/knowledge/tasks/notification-policy`);
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '提醒策略加载失败');
                knowledgeTaskNotificationPolicy.value = { ...knowledgeTaskNotificationPolicy.value, ...(data.policy || {}) };
            } catch (error) {
                knowledgeTaskPolicyMessage.value = error.message || '提醒策略加载失败';
            } finally {
                knowledgeTaskPolicyLoading.value = false;
            }
        }

        async function openKnowledgeTaskPolicy() {
            knowledgeTaskPolicyMessage.value = '';
            showKnowledgeTaskPolicyModal.value = true;
            await loadKnowledgeTaskNotificationPolicy();
        }

        async function saveKnowledgeTaskNotificationPolicy() {
            if (!canManageKnowledgeTasks.value) return;
            knowledgeTaskPolicySaving.value = true;
            knowledgeTaskPolicyMessage.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/knowledge/tasks/notification-policy`, {
                    method: 'PUT', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(knowledgeTaskNotificationPolicy.value),
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '提醒策略保存失败');
                knowledgeTaskNotificationPolicy.value = { ...knowledgeTaskNotificationPolicy.value, ...(data.policy || {}) };
                knowledgeTaskPolicyMessage.value = data.message || '自动提醒策略已保存';
            } catch (error) {
                knowledgeTaskPolicyMessage.value = error.message || '提醒策略保存失败';
            } finally {
                knowledgeTaskPolicySaving.value = false;
            }
        }

        async function scanKnowledgeTaskNotifications() {
            if (!canManageKnowledgeTasks.value) return;
            knowledgeTaskPolicyScanning.value = true;
            knowledgeTaskPolicyMessage.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/knowledge/tasks/notification-policy/scan`, { method: 'POST' });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '提醒扫描失败');
                knowledgeTaskPolicyMessage.value = data.result?.deferred_by_quiet_hours
                    ? `${data.message}，将在静默时段结束后投递`
                    : data.message;
                if (data.result?.job_count && can('job.read')) await loadProcessingJobs({ keepSelection: false, silent: true });
            } catch (error) {
                knowledgeTaskPolicyMessage.value = error.message || '提醒扫描失败';
            } finally {
                knowledgeTaskPolicyScanning.value = false;
            }
        }

        function openKnowledgeTaskModal() {
            knowledgeTaskDraft.value = {
                title: '', description: '', task_type: 'manual', priority: 'medium',
                assignee_user_id: '', due_at: '', source_type: 'manual', source_key: '',
                dedupe_key: '', handover_id: '',
            };
            knowledgeTaskMessage.value = '';
            showKnowledgeTaskModal.value = true;
        }

        function openFeishuReviewTask(candidate) {
            const candidateId = String(candidate?.candidate_id || '');
            knowledgeTaskDraft.value = {
                title: `审核飞书知识候选：${candidate?.title || candidateId || '待确认内容'}`,
                description: candidate?.reasons?.join('；') || candidate?.last_error || '请核对来源、分类、敏感性和知识价值后决定是否入库。',
                task_type: 'feishu_review', priority: 'high', assignee_user_id: '', due_at: '',
                source_type: 'feishu_candidate', source_key: candidateId,
                dedupe_key: candidateId ? `feishu_review:${candidateId}` : '', handover_id: '',
            };
            knowledgeTaskMessage.value = '';
            showKnowledgeTaskModal.value = true;
        }

        function openHandoverGapTask(record) {
            const handoverId = String(record?.id || '');
            const departingUserId = String(record?.departing_user_id || '');
            const departingName = String(record?.name || '').trim().toLowerCase();
            // 负责人在服务端按交接记录默认取离职人员；无指派权限的成员不预填，避免越权指派
            const assignee = canManageKnowledgeTasks.value
                ? (projectMembers.value.find(item => departingUserId && String(item.user_id || '') === departingUserId)
                    || projectMembers.value.find(item => [item.email, item.display_name].some(value => String(value || '').trim().toLowerCase() === departingName)))
                : null;
            knowledgeTaskDraft.value = {
                title: `补充${record?.name || '成员'}的交接资料`,
                description: `请补齐交接资料、未关闭风险及责任说明，并由接替人核对。当前风险：${(record?.risks || []).join('；') || '待核对'}`,
                task_type: 'handover_gap', priority: 'high', assignee_user_id: assignee?.user_id || '',
                due_at: record?.due_date ? `${record.due_date}T18:00` : '',
                source_type: 'handover', source_key: handoverId,
                dedupe_key: handoverId ? `handover_gap:${handoverId}` : '', handover_id: handoverId,
            };
            knowledgeTaskMessage.value = '';
            showKnowledgeTaskModal.value = true;
        }

        async function loadKnowledgeTasks(options = {}) {
            knowledgeTasksLoading.value = true;
            knowledgeTasksError.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/knowledge/tasks?limit=200`);
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '知识任务加载失败');
                knowledgeTasks.value = data.tasks || [];
                knowledgeTaskSummary.value = data.summary || {};
                if (selectedKnowledgeTask.value) {
                    const latest = knowledgeTasks.value.find(item => item.task_id === selectedKnowledgeTask.value.task_id);
                    if (latest && options.keepSelection !== false) await openKnowledgeTask(latest);
                    else selectedKnowledgeTask.value = null;
                }
            } catch (error) {
                knowledgeTasksError.value = error.message || '知识任务加载失败';
            } finally {
                knowledgeTasksLoading.value = false;
            }
        }

        async function openKnowledgeTask(task) {
            if (!task?.task_id) return;
            knowledgeTaskDetailLoading.value = true;
            knowledgeTaskMessage.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/knowledge/tasks/${encodeURIComponent(task.task_id)}`);
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '任务详情加载失败');
                selectedKnowledgeTask.value = data.task;
                knowledgeTaskEvidence.value = data.task.completion_evidence || '';
                knowledgeTaskEvidenceUploadMessage.value = '';
                knowledgeTaskAssignee.value = data.task.assignee_user_id || '';
                knowledgeTaskNote.value = '';
            } catch (error) {
                knowledgeTaskMessage.value = error.message || '任务详情加载失败';
            } finally {
                knowledgeTaskDetailLoading.value = false;
            }
        }

        async function createKnowledgeTask() {
            if (!knowledgeTaskDraft.value.title.trim()) {
                knowledgeTaskMessage.value = '请填写任务标题';
                return;
            }
            knowledgeTaskSaving.value = true;
            knowledgeTaskMessage.value = '';
            try {
                const payload = { ...knowledgeTaskDraft.value };
                if (payload.due_at) payload.due_at = new Date(payload.due_at).toISOString();
                const res = await apiFetch(`${API_BASE}/knowledge/tasks`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '任务创建失败');
                showKnowledgeTaskModal.value = false;
                knowledgeTaskMessage.value = data.message || '任务已创建';
                await loadKnowledgeTasks({ keepSelection: false });
                await openKnowledgeTask(data.task);
            } catch (error) {
                knowledgeTaskMessage.value = error.message || '任务创建失败';
            } finally {
                knowledgeTaskSaving.value = false;
            }
        }

        function openKnowledgeTaskCancel() {
            if (!selectedKnowledgeTask.value?.task_id) return;
            knowledgeTaskCancelReason.value = knowledgeTaskNote.value.trim();
            knowledgeTaskCancelMessage.value = '';
            showKnowledgeTaskCancelModal.value = true;
        }

        function closeKnowledgeTaskCancel() {
            if (knowledgeTaskActionLoading.value === 'cancel') return;
            showKnowledgeTaskCancelModal.value = false;
            knowledgeTaskCancelReason.value = '';
            knowledgeTaskCancelMessage.value = '';
        }

        async function confirmKnowledgeTaskCancel() {
            if (!knowledgeTaskCancelReason.value.trim()) {
                knowledgeTaskCancelMessage.value = '请填写取消原因';
                return;
            }
            const success = await actOnKnowledgeTask('cancel', {
                note: knowledgeTaskCancelReason.value.trim(),
                messageTarget: 'cancel',
            });
            if (success) {
                showKnowledgeTaskCancelModal.value = false;
                knowledgeTaskCancelReason.value = '';
                knowledgeTaskCancelMessage.value = '';
            }
        }

        async function actOnKnowledgeTask(action, options = {}) {
            const task = selectedKnowledgeTask.value;
            if (!task?.task_id || knowledgeTaskActionLoading.value) return false;
            const actionNote = String(options.note ?? knowledgeTaskNote.value).trim();
            const setActionMessage = (message) => {
                if (options.messageTarget === 'cancel') knowledgeTaskCancelMessage.value = message;
                else knowledgeTaskMessage.value = message;
            };
            if (action === 'submit' && !knowledgeTaskEvidence.value.trim()) {
                setActionMessage('提交验收前请填写完成证据');
                return false;
            }
            if (['return', 'cancel'].includes(action) && !actionNote) {
                setActionMessage(action === 'return' ? '请填写退回原因' : '请填写取消原因');
                return false;
            }
            knowledgeTaskActionLoading.value = action;
            knowledgeTaskMessage.value = '';
            if (options.messageTarget === 'cancel') knowledgeTaskCancelMessage.value = '';
            try {
                const payload = {
                    action,
                    note: actionNote,
                    evidence: knowledgeTaskEvidence.value.trim(),
                    assignee_user_id: knowledgeTaskAssignee.value,
                };
                const res = await apiFetch(`${API_BASE}/knowledge/tasks/${encodeURIComponent(task.task_id)}/actions`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '任务更新失败');
                selectedKnowledgeTask.value = data.task;
                knowledgeTaskNote.value = '';
                await loadKnowledgeTasks();
                if (data.task?.writeback?.applied) {
                    if (data.task.writeback.type === 'handover_item_resubmit') await loadHandoverRecords();
                    if (['asset_review_confirm', 'asset_version_publish'].includes(data.task.writeback.type)) await loadKnowledgeAssets();
                }
                knowledgeTaskMessage.value = data.message || '任务状态已更新';
                return true;
            } catch (error) {
                setActionMessage(error.message || '任务更新失败');
                return false;
            } finally {
                knowledgeTaskActionLoading.value = '';
            }
        }

        function recordKnowledgeTaskUpload(context, filenames) {
            if (!context?.taskId || context.taskId !== selectedKnowledgeTask.value?.task_id) return;
            const uniqueNames = [...new Set((filenames || []).filter(Boolean))];
            if (!uniqueNames.length) return;
            const references = uniqueNames.map(name => `已上传任务资料：${name}（知识库处理中）`);
            knowledgeTaskEvidence.value = [knowledgeTaskEvidence.value.trim(), ...references]
                .filter(Boolean).join('\n');
            knowledgeTaskEvidenceUploadMessage.value = `已受理 ${uniqueNames.length} 份任务资料并写入完成证据；请在处理任务确认入库成功后再提交验收。`;
        }

        async function openKnowledgeTaskUploadModal() {
            const task = selectedKnowledgeTask.value;
            if (!task?.task_id || task.status !== 'in_progress' || !knowledgeTaskCan('submit')) return;
            resetKbUploadForm();
            uploadMode.value = 'create';
            knowledgeTaskUploadContext.value = {
                taskId: task.task_id,
                title: task.title || '当前知识任务',
            };
            knowledgeTaskEvidenceUploadMessage.value = '';
            showUploadModal.value = true;
        }

        async function createOnboardingGapTasks() {
            if (!onboardingData.value?.gaps?.length) {
                onboardingTaskMessage.value = '当前岗位没有待补知识';
                return;
            }
            onboardingTaskMessage.value = '正在生成知识补充任务…';
            try {
                const res = await apiFetch(`${API_BASE}/knowledge/tasks/onboarding-gaps`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        role: onboardingRole.value, due_days: 7,
                        assignee_user_id: can('onboarding.manage') ? onboardingTargetUserId.value : '',
                    }),
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '任务生成失败');
                onboardingTaskMessage.value = data.message;
                await loadKnowledgeTasks({ keepSelection: false });
            } catch (error) {
                onboardingTaskMessage.value = error.message || '任务生成失败';
            }
        }

        async function notifyKnowledgeTask() {
            const task = selectedKnowledgeTask.value;
            if (!task?.task_id || knowledgeTaskActionLoading.value) return;
            knowledgeTaskActionLoading.value = 'notify';
            knowledgeTaskMessage.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/knowledge/tasks/${encodeURIComponent(task.task_id)}/notify`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ chat_id: knowledgeTaskChatId.value.trim() || feishuDefaultChatId.value }),
                });
                const data = await res.json().catch(() => ({}));
                knowledgeTaskMessage.value = data.message || (res.ok ? '提醒已发送' : '提醒发送失败');
                await openKnowledgeTask(task);
            } catch (error) {
                knowledgeTaskMessage.value = '提醒发送失败，任务数据未受影响：' + error.message;
            } finally {
                knowledgeTaskActionLoading.value = '';
            }
        }

        async function submitResignation() {
            const departingName = currentUser.value?.display_name || currentUser.value?.email || '';
            if (!currentUser.value?.user_id || !departingName) {
                alert('未读取到当前登录账号，请重新登录后提交');
                return;
            }
            resignSubmitting.value = true;
            resignResult.value = null;
            resignError.value = '';
            activeHandoverStep.value = 0;
            const handoverTimer = setInterval(() => {
                activeHandoverStep.value = Math.min(handoverFlowSteps.length - 1, activeHandoverStep.value + 1);
            }, 480);
            try {
                const form = new FormData();
                form.append('name', departingName);
                form.append('role', currentUser.value?.duty || '');
                const recipientMember = projectMembers.value.find(
                    member => member.user_id === resignRecipient.value,
                );
                form.append('recipient', recipientMember?.display_name || '');
                form.append('recipient_user_id', recipientMember?.user_id || '');
                form.append('due_date', resignDueDate.value);
                for (const file of resignFiles.value) {
                    form.append('files', file);
                }
                const res = await apiFetch(`${API_BASE}/resignation/submit`, { method: 'POST', body: form });
                const data = await res.json().catch(() => ({}));
                if (res.ok && data.success) {
                    resignResult.value = data;
                    resignError.value = '';
                    await loadGroupedDocs();
                    await loadStats();
                    await loadHandoverRecords();
                    applyHandoverStage(data.handover);
                } else {
                    resignError.value = data.detail || data.message || '提交失败';
                }
            } catch (e) {
                resignError.value = '请求失败: ' + e.message;
            } finally {
                clearInterval(handoverTimer);
                resignSubmitting.value = false;
            }
        }

        function valueRequiresRepair(value) {
            if (value === true) return true;
            if (!value) return false;
            if (Array.isArray(value)) return value.length > 0;
            if (typeof value === 'object') return Object.keys(value).length > 0;
            return !['false', 'healthy', 'ready', 'none', '0'].includes(String(value).trim().toLowerCase());
        }

        const resignResultNeedsRepair = computed(() => {
            const result = resignResult.value;
            return Boolean(result?.partial_success || valueRequiresRepair(result?.repair_required));
        });

        const resignRepairDetails = computed(() => {
            const result = resignResult.value || {};
            const details = [];
            (result.failed_files || []).forEach(file => details.push(`${file.filename || '交接文件'}：${file.reason || '未处理'}`));
            (result.sync_warnings || []).forEach(item => {
                if (typeof item === 'string') {
                    details.push(item);
                    return;
                }
                if (item && typeof item === 'object') {
                    const name = item.filename || item.asset_id || '交接文档';
                    const projections = Array.isArray(item.projections)
                        ? item.projections.join('、')
                        : (item.reason || item.message || '派生同步待修复');
                    details.push(`${name}：${projections}`);
                }
            });
            if (Array.isArray(result.repair_required)) {
                result.repair_required.forEach(item => {
                    if (!item || typeof item !== 'object') return;
                    const name = item.filename || item.asset_id || '交接文档';
                    const projections = Array.isArray(item.projections)
                        ? item.projections.join('、')
                        : '派生同步待修复';
                    details.push(`${name}：${projections}`);
                });
            }
            if (typeof result.repair_required === 'string') details.push(result.repair_required);
            if (result.repair_message) details.push(result.repair_message);
            if (result.repair_action) details.push(result.repair_action);
            if (!details.length && resignResultNeedsRepair.value) details.push('部分交接文档或派生知识尚未完成，请在知识库中查看同步状态并修复。');
            return [...new Set(details.filter(Boolean))];
        });

        // ── 系统设置 ──

        async function saveLLMSettings() {
            savingKey.value = true;
            saveKeyMsg.value = '';
            llmTestMsg.value = '';
            try {
                const form = new FormData();
                if (apiKey.value.trim()) form.append('api_key', apiKey.value.trim());
                form.append('api_base_url', apiBaseUrl.value);
                form.append('llm_model', llmModel.value);
                form.append('llm_provider', llmProvider.value);
                const res = await apiFetch(`${API_BASE}/settings/llm`, { method: 'POST', body: form });
                const data = await res.json();
                if (data.success) {
                    apiKeyConfigured.value = !!data.api_key_configured;
                    apiKey.value = '';
                    apiBaseUrl.value = data.api_base_url || apiBaseUrl.value;
                    llmModel.value = data.llm_model || llmModel.value;
                    saveKeyMsg.value = '成功：配置已保存，正在测试模型连接...';
                    await checkHealth();
                    const testResult = await testLLMConnection(false);
                    saveKeyMsg.value = testResult.success
                        ? '成功：配置已保存，模型连接正常'
                        : '已保存，但模型连接测试失败';
                } else {
                    saveKeyMsg.value = '失败：' + (data.detail || '保存失败');
                }
            } catch (e) {
                saveKeyMsg.value = '失败：保存失败: ' + e.message;
            } finally {
                savingKey.value = false;
            }
        }

        async function testLLMConnection(showProgress = true) {
            testingLLM.value = true;
            llmTestLatency.value = null;
            if (showProgress) {
                llmTestMsg.value = '正在测试模型连接...';
            }
            try {
                const startedAt = performance.now();
                const res = await apiFetch(`${API_BASE}/settings/llm/test`, { method: 'POST' });
                const data = await res.json();
                llmTestLatency.value = Math.round(performance.now() - startedAt);
                apiKeyConfigured.value = !!data.api_key_configured;
                llmTestOk.value = !!data.success;
                llmTestMsg.value = data.success
                    ? `连接正常：${data.llm_model || llmModel.value} 已响应，耗时 ${llmTestLatency.value} ms`
                    : `连接失败：${data.detail || data.message || '请检查 API Key、API 地址和模型名称'}`;
                await checkHealth();
                return data;
            } catch (e) {
                llmTestOk.value = false;
                llmTestMsg.value = '连接失败：' + e.message;
                return { success: false, detail: e.message };
            } finally {
                testingLLM.value = false;
            }
        }

        async function saveNetworkSettings() {
            savingNetwork.value = true;
            networkMsg.value = '';
            try {
                const form = new FormData();
                form.append('proxy_enabled', String(proxyEnabled.value));
                form.append('proxy_url', proxyUrl.value);
                form.append('verify_ssl', String(verifySsl.value));
                const res = await apiFetch(`${API_BASE}/settings/network`, { method: 'POST', body: form });
                const data = await res.json();
                networkMsg.value = data.success ? '成功：' + data.message : '失败：' + (data.detail || '保存失败');
            } catch (e) {
                networkMsg.value = '失败：保存失败: ' + e.message;
            } finally {
                savingNetwork.value = false;
            }
        }

        async function resetKnowledgeLibrary() {
            const confirmed = window.confirm('确定清空所有知识库数据吗？当前、待复审、已失效和已撤销的知识资产都会一并清除，并永久删除知识来源、原件、文档、索引、图谱、处理任务、知识任务、新人赋能学习计划、离职交接记录，以及飞书已保存的历史消息、候选、飞书知识和资源文件。飞书采集群配置、账号、项目成员、分类、删除墓碑与安全审计会保留，且不会创建备份。');
            if (!confirmed) return;
            resettingKnowledgeLibrary.value = true;
            knowledgeLibraryResetMessage.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/settings/knowledge-library/reset`, { method: 'POST' });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '知识库清空失败');
                knowledgeLibraryResetMessage.value = `成功：${data.message || '知识库已清空'}`;
                selectedKnowledgeAsset.value = null;
                selectedKnowledgeTask.value = null;
                selectedProcessingJob.value = null;
                selectedHandover.value = null;
                onboardingData.value = null;
                showHandoverModal.value = false;
                resignResult.value = null;
                feishuContentMessages.value = [];
                feishuContentAssets.value = [];
                feishuContentTotal.value = 0;
                await Promise.all([
                    loadDocuments(), loadStats(), loadGraph(), loadGroupedDocs(), loadKnowledgeAssets(),
                    loadHandoverRecords(),
                    can('task.read') ? loadKnowledgeTasks({ keepSelection: false }) : Promise.resolve(),
                    can('job.read') ? loadProcessingJobs({ keepSelection: false, silent: true }) : Promise.resolve(),
                    can('feishu.read') ? loadFeishuWorkspace() : Promise.resolve(),
                ]);
                if (tab.value === 'graph') renderGraphSoon();
            } catch (error) {
                knowledgeLibraryResetMessage.value = `失败：${error.message || '知识库清空失败'}`;
            } finally {
                resettingKnowledgeLibrary.value = false;
            }
        }

        function governanceAuditLabel(action) {
            const labels = {
                'access.knowledge_answer': '访问知识问答',
                'access.document_preview': '预览知识文档',
                'access.knowledge_asset': '查看知识资产',
                'access.knowledge_graph': '查看知识图谱',
                'access.feishu_resource': '查看飞书资源',
                'governance.policy_updated': '更新治理策略',
                'governance.backup_created': '创建数据库备份',
                'governance.retention_applied': '执行数据清理',
                'governance.audit_exported': '导出审计记录',
                'governance.target_exported': '导出定向数据',
                'governance.target_deleted': '删除定向数据',
                'governance.target_files_removed': '清除定向原件',
                'governance.daily_jobs_scheduled': '安排每日治理任务',
                'knowledge.library_reset': '清空项目知识库数据',
                'iam.permission_denied': '权限访问被拒绝',
                'iam.csrf_denied': '安全校验被拒绝',
            };
            return labels[action] || action || '系统操作';
        }

        function formatFileSize(bytes) {
            const value = Number(bytes || 0);
            if (value < 1024) return `${value} B`;
            if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
            return `${(value / 1024 / 1024).toFixed(1)} MB`;
        }

        async function loadGovernanceAudit() {
            const params = new URLSearchParams({
                action: governanceAuditAction.value,
                keyword: governanceAuditKeyword.value.trim(),
                page: '1', page_size: '12',
            });
            const res = await apiFetch(`${API_BASE}/governance/audit?${params}`);
            const data = await res.json().catch(() => ({}));
            if (!res.ok || !data.success) throw new Error(data.detail || '审计记录加载失败');
            governanceAudit.value = data.items || [];
            governanceAuditTotal.value = data.total || 0;
        }

        function operationsStatusLabel(status) {
            return {
                healthy: '正常', degraded: '需处理', warning: '需关注',
                collecting: '采集中', unconfigured: '未配置',
            }[status] || '未知';
        }

        function operationsMetricValue(metric) {
            if (metric?.value === null || metric?.value === undefined) return '样本不足';
            const value = Number(metric.value);
            if (metric.unit === '%') return `${value.toFixed(2)}%`;
            return `${value.toLocaleString()} ${metric.unit || ''}`.trim();
        }

        async function loadOperationsOverview({ preserveMessage = false } = {}) {
            if (!can('settings.manage')) return;
            operationsLoading.value = true;
            if (!preserveMessage) operationsMessage.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/ops/overview?window=${encodeURIComponent(operationsWindow.value)}`);
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '运行监控加载失败');
                operationsOverview.value = data;
            } catch (error) {
                operationsMessage.value = `失败：${error.message || '运行监控加载失败'}`;
            } finally {
                operationsLoading.value = false;
            }
        }

        async function runCapacityValidation() {
            operationsActionLoading.value = true;
            operationsMessage.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/ops/capacity/validate`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message_count: 10000 }),
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '容量验证提交失败');
                operationsMessage.value = `容量验证已进入处理任务（${data.job?.job_id || ''}）`;
                await loadProcessingJobs({ silent: true });
                setTimeout(() => loadOperationsOverview({ preserveMessage: true }), 1800);
            } catch (error) {
                operationsMessage.value = `失败：${error.message || '容量验证提交失败'}`;
            } finally {
                operationsActionLoading.value = false;
            }
        }

        async function createMaintenanceWindow() {
            const draft = maintenanceDraft.value;
            if (!draft.title.trim() || !draft.starts_at || !draft.ends_at) {
                operationsMessage.value = '失败：请填写维护标题、开始和结束时间';
                return;
            }
            operationsActionLoading.value = true;
            operationsMessage.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/ops/maintenance`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        title: draft.title.trim(), reason: draft.reason.trim(),
                        starts_at: new Date(draft.starts_at).toISOString(),
                        ends_at: new Date(draft.ends_at).toISOString(),
                    }),
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '维护窗口登记失败');
                maintenanceDraft.value = { title: '', reason: '', starts_at: '', ends_at: '' };
                operationsMessage.value = '计划维护窗口已登记，将从工作时段可用性中单独排除';
                await loadOperationsOverview({ preserveMessage: true });
            } catch (error) {
                operationsMessage.value = `失败：${error.message || '维护窗口登记失败'}`;
            } finally {
                operationsActionLoading.value = false;
            }
        }

        async function cancelMaintenanceWindow(windowId) {
            operationsActionLoading.value = true;
            operationsMessage.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/ops/maintenance/${encodeURIComponent(windowId)}`, { method: 'DELETE' });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '维护窗口取消失败');
                operationsMessage.value = '计划维护窗口已取消';
                await loadOperationsOverview({ preserveMessage: true });
            } catch (error) {
                operationsMessage.value = `失败：${error.message || '维护窗口取消失败'}`;
            } finally {
                operationsActionLoading.value = false;
            }
        }

        async function runRecoveryDrill() {
            operationsActionLoading.value = true;
            operationsMessage.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/ops/recovery-drills`, { method: 'POST' });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '恢复演练提交失败');
                operationsMessage.value = `恢复演练已进入处理任务（${data.job?.job_id || ''}）`;
                await loadProcessingJobs({ silent: true });
                setTimeout(() => loadOperationsOverview({ preserveMessage: true }), 1800);
            } catch (error) {
                operationsMessage.value = `失败：${error.message || '恢复演练提交失败'}`;
            } finally {
                operationsActionLoading.value = false;
            }
        }

        async function loadDataGovernance({ preserveMessage = false } = {}) {
            if (!can('settings.manage')) return;
            governanceLoading.value = true;
            if (!preserveMessage) governanceMessage.value = '';
            try {
                const [policyRes, previewRes, backupRes, scheduleRes] = await Promise.all([
                    apiFetch(`${API_BASE}/governance/policy`),
                    apiFetch(`${API_BASE}/governance/retention/preview`),
                    apiFetch(`${API_BASE}/governance/backups`),
                    apiFetch(`${API_BASE}/governance/schedule`),
                ]);
                const [policyData, previewData, backupData, scheduleData] = await Promise.all([
                    policyRes.json(), previewRes.json(), backupRes.json(), scheduleRes.json(),
                ]);
                if (!policyRes.ok) throw new Error(policyData.detail || '治理策略加载失败');
                if (!previewRes.ok) throw new Error(previewData.detail || '清理预演加载失败');
                if (!backupRes.ok) throw new Error(backupData.detail || '备份列表加载失败');
                if (!scheduleRes.ok) throw new Error(scheduleData.detail || '治理调度加载失败');
                governancePolicy.value = { ...governancePolicy.value, ...(policyData.policy || {}) };
                governancePreview.value = previewData.preview || {};
                governanceBackups.value = backupData.backups || [];
                governanceSchedule.value = {
                    next_scheduled_at: scheduleData.next_scheduled_at || '',
                    latest_intents: scheduleData.latest_intents || [],
                };
                await loadGovernanceAudit();
            } catch (error) {
                governanceMessage.value = error.message || '数据治理信息加载失败';
            } finally {
                governanceLoading.value = false;
            }
        }

        async function saveDataGovernancePolicy() {
            governanceActionLoading.value = 'policy';
            governanceMessage.value = '';
            try {
                const payload = {
                    raw_message_retention_days: Number(governancePolicy.value.raw_message_retention_days),
                    access_audit_retention_days: Number(governancePolicy.value.access_audit_retention_days),
                    processing_history_retention_days: Number(governancePolicy.value.processing_history_retention_days),
                    backup_retention_count: Number(governancePolicy.value.backup_retention_count),
                    default_sensitivity: governancePolicy.value.default_sensitivity,
                    daily_backup_enabled: Boolean(governancePolicy.value.daily_backup_enabled),
                    daily_retention_enabled: Boolean(governancePolicy.value.daily_retention_enabled),
                    schedule_time: governancePolicy.value.schedule_time,
                    timezone: governancePolicy.value.timezone,
                };
                const res = await apiFetch(`${API_BASE}/governance/policy`, {
                    method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '策略保存失败');
                governancePolicy.value = data.policy;
                governanceMessage.value = data.message || '数据治理策略已保存';
                await loadDataGovernance({ preserveMessage: true });
            } catch (error) {
                governanceMessage.value = `失败：${error.message || '策略保存失败'}`;
            } finally {
                governanceActionLoading.value = '';
            }
        }

        async function createDataBackup() {
            governanceActionLoading.value = 'backup';
            governanceMessage.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/governance/backups`, { method: 'POST' });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '备份任务创建失败');
                governanceMessage.value = `${data.message}（${data.job?.job_id || ''}）`;
                await loadProcessingJobs({ silent: true });
                setTimeout(() => loadDataGovernance({ preserveMessage: true }), 1200);
            } catch (error) {
                governanceMessage.value = `失败：${error.message || '备份任务创建失败'}`;
            } finally {
                governanceActionLoading.value = '';
            }
        }

        function governanceScheduleOperationLabel(operation) {
            return operation === 'data_retention' ? '保留清理' : '数据库备份';
        }

        async function scanDataGovernanceSchedule() {
            governanceActionLoading.value = 'schedule-scan';
            governanceMessage.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/governance/schedule/scan`, { method: 'POST' });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '治理调度扫描失败');
                governanceMessage.value = data.message || '治理调度扫描完成';
                await loadProcessingJobs({ silent: true });
                await loadDataGovernance({ preserveMessage: true });
            } catch (error) {
                governanceMessage.value = `失败：${error.message || '治理调度扫描失败'}`;
            } finally {
                governanceActionLoading.value = '';
            }
        }

        async function applyDataRetention() {
            if (governanceConfirm.value !== '执行数据清理') {
                governanceMessage.value = '请输入完整确认短语：执行数据清理';
                return;
            }
            governanceActionLoading.value = 'retention';
            governanceMessage.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/governance/retention/apply`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ confirmation: governanceConfirm.value }),
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '清理任务创建失败');
                governanceConfirm.value = '';
                governanceMessage.value = `${data.message}（${data.job?.job_id || ''}）`;
                await loadProcessingJobs({ silent: true });
                setTimeout(() => loadDataGovernance({ preserveMessage: true }), 1200);
            } catch (error) {
                governanceMessage.value = `失败：${error.message || '清理任务创建失败'}`;
            } finally {
                governanceActionLoading.value = '';
            }
        }

        async function exportGovernanceAudit() {
            governanceActionLoading.value = 'export';
            governanceMessage.value = '';
            try {
                const params = new URLSearchParams({
                    action: governanceAuditAction.value,
                    keyword: governanceAuditKeyword.value.trim(),
                });
                const res = await apiFetch(`${API_BASE}/governance/audit/export?${params}`);
                if (!res.ok) {
                    const data = await res.json().catch(() => ({}));
                    throw new Error(data.detail || '审计导出失败');
                }
                const blob = await res.blob();
                const link = document.createElement('a');
                link.href = URL.createObjectURL(blob);
                link.download = `audit-${new Date().toISOString().slice(0, 10)}.csv`;
                link.click();
                URL.revokeObjectURL(link.href);
                governanceMessage.value = '审计记录已导出';
                await loadGovernanceAudit();
            } catch (error) {
                governanceMessage.value = `失败：${error.message || '审计导出失败'}`;
            } finally {
                governanceActionLoading.value = '';
            }
        }

        function governanceTargetParams() {
            return new URLSearchParams({
                target_type: governanceTarget.value.target_type,
                target_id: governanceTarget.value.target_id.trim(),
            });
        }

        async function previewGovernanceTarget() {
            if (!governanceTarget.value.target_id.trim()) {
                governanceMessage.value = '失败：请输入来源 ID、成员 ID 或资产 ID';
                return;
            }
            governanceActionLoading.value = 'target-preview';
            governanceMessage.value = '';
            governanceTargetPreview.value = null;
            governanceTarget.value.confirmation = '';
            try {
                const res = await apiFetch(`${API_BASE}/governance/targets/preview?${governanceTargetParams()}`);
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '定向范围预演失败');
                governanceTargetPreview.value = data.preview || null;
                governanceMessage.value = '定向范围已重新核对；建议删除前先导出留档';
            } catch (error) {
                governanceMessage.value = `失败：${error.message || '定向范围预演失败'}`;
            } finally {
                governanceActionLoading.value = '';
            }
        }

        async function exportGovernanceTarget() {
            if (!governanceTargetPreview.value) {
                governanceMessage.value = '失败：请先预演定向范围';
                return;
            }
            governanceActionLoading.value = 'target-export';
            governanceMessage.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/governance/targets/export?${governanceTargetParams()}`);
                if (!res.ok) {
                    const data = await res.json().catch(() => ({}));
                    throw new Error(data.detail || '定向数据导出失败');
                }
                const blob = await res.blob();
                const link = document.createElement('a');
                link.href = URL.createObjectURL(blob);
                link.download = `governance-${governanceTarget.value.target_type}-${new Date().toISOString().slice(0, 10)}.json`;
                link.click();
                URL.revokeObjectURL(link.href);
                governanceMessage.value = '定向数据已导出并记录审计';
                await loadGovernanceAudit();
            } catch (error) {
                governanceMessage.value = `失败：${error.message || '定向数据导出失败'}`;
            } finally {
                governanceActionLoading.value = '';
            }
        }

        async function deleteGovernanceTarget() {
            const preview = governanceTargetPreview.value;
            const target = governanceTarget.value;
            if (!preview || preview.target_type !== target.target_type || preview.target_id !== target.target_id.trim()) {
                governanceMessage.value = '失败：目标已变化，请重新预演';
                return;
            }
            if (target.confirmation !== '永久删除定向数据') {
                governanceMessage.value = '失败：请输入完整确认短语：永久删除定向数据';
                return;
            }
            governanceActionLoading.value = 'target-delete';
            governanceMessage.value = '';
            try {
                const res = await apiFetch(`${API_BASE}/governance/targets/delete`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        target_type: target.target_type, target_id: target.target_id.trim(),
                        confirmation: target.confirmation,
                    }),
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.detail || '定向删除任务创建失败');
                governanceTarget.value = { target_type: target.target_type, target_id: '', confirmation: '' };
                governanceTargetPreview.value = null;
                governanceMessage.value = `${data.message}（${data.job?.job_id || ''}）`;
                await loadProcessingJobs({ silent: true });
                setTimeout(() => loadDataGovernance({ preserveMessage: true }), 1200);
            } catch (error) {
                governanceMessage.value = `失败：${error.message || '定向删除任务创建失败'}`;
            } finally {
                governanceActionLoading.value = '';
            }
        }

        async function checkTunnelStatus() {
            try {
                const res = await apiFetch(`${API_BASE}/tunnel/status`);
                const data = await res.json();
                if (data.success) {
                    tunnelRunning.value = data.running;
                    tunnelUrl.value = data.url || '';
                }
            } catch (e) { /* 忽略 */ }
        }

        async function startTunnel() {
            tunnelLoading.value = true;
            tunnelMsg.value = '';
            try {
                const form = new FormData();
                const res = await apiFetch(`${API_BASE}/tunnel/start`, { method: 'POST', body: form });
                const data = await res.json();
                tunnelMsg.value = data.success ? (data.url ? '成功：' + data.message : '启动中：' + data.message) : '失败：' + data.message;
                if (data.success) {
                    tunnelRunning.value = true;
                    if (data.url) tunnelUrl.value = data.url;
                }
            } catch (e) {
                tunnelMsg.value = '失败：请求失败: ' + e.message;
            } finally {
                tunnelLoading.value = false;
            }
        }

        async function stopTunnel() {
            tunnelLoading.value = true;
            tunnelMsg.value = '';
            try {
                const form = new FormData();
                const res = await apiFetch(`${API_BASE}/tunnel/stop`, { method: 'POST', body: form });
                const data = await res.json();
                tunnelMsg.value = data.success ? '成功：' + data.message : '失败：' + data.message;
                if (data.success) {
                    tunnelRunning.value = false;
                    tunnelUrl.value = '';
                }
            } catch (e) {
                tunnelMsg.value = '失败：请求失败: ' + e.message;
            } finally {
                tunnelLoading.value = false;
            }
        }

        async function restartServer() {
            if (!confirm('确认重启服务？当前对话会中断，服务将在几秒后恢复。')) return;
            saveKeyMsg.value = '正在重启服务...';
            try { await apiFetch(`${API_BASE}/restart`, { method: 'POST' }); } catch (e) {}
            saveKeyMsg.value = '等待服务重启...';
            for (let i = 0; i < 30; i++) {
                await new Promise(r => setTimeout(r, 1000));
                try {
                    const res = await apiFetch(`${API_BASE}/health`);
                    if (res.ok) {
                        const data = await res.json();
                        storageReady.value = data.storage_ok === true;
                        if (!storageReady.value) continue;
                        llmReady.value = data.llm_configured || false;
                        saveKeyMsg.value = '服务已重启';
                        await loadDocuments();
                        await loadStats();
                        return;
                    }
                } catch (e) {}
            }
            saveKeyMsg.value = '重启超时，请手动刷新页面';
        }

        async function checkHealth() {
            try {
                const res = await apiFetch(`${API_BASE}/health`);
                if (res.ok) {
                    const data = await res.json();
                    storageReady.value = data.storage_ok === true;
                    llmReady.value = data.llm_configured || false;
                    apiKeyConfigured.value = data.api_key_configured || false;
                    feishuConfigured.value = data.feishu_configured || false;
                    // 回填非敏感配置；API Key / App Secret 只允许写入，不从服务端读回。
                    feishuAppId.value = data.feishu_app_id || '';
                    feishuDefaultChatId.value = data.feishu_default_chat_id || '';
                    apiBaseUrl.value = data.api_base_url || 'https://api.openai.com/v1';
                    llmModel.value = data.llm_model || 'gpt-4o-mini';
                    // 根据保存的 API 地址推断提供商
                    const savedUrl = data.api_base_url || '';
                    if (savedUrl.includes('minimax')) llmProvider.value = 'minimax';
                    else if (savedUrl.includes('deepseek')) llmProvider.value = 'deepseek';
                    else if (savedUrl.includes('api.openai')) llmProvider.value = 'openai';
                    else if (savedUrl) llmProvider.value = 'custom';
                    proxyEnabled.value = data.proxy_enabled || false;
                    proxyUrl.value = data.proxy_url || '';
                    verifySsl.value = data.verify_ssl !== false;
                }
            } catch (e) { console.error('健康检查失败:', e); }
        }

        // ── 初始化 ──

        watch(tab, (nextTab) => {
            mobileNavOpen.value = false;
            if (nextTab === 'knowledge' && !knowledgeAssetsLoading.value) loadKnowledgeAssets();
            if (nextTab === 'graph') renderGraphSoon();
            if (nextTab === 'onboarding') {
                if (!onboardingData.value && !onboardingLoading.value) loadOnboardingGuide();
                if (can('onboarding.manage') && !supervisedOnboardingPlansLoading.value) loadSupervisedOnboardingPlans();
            }
            if (nextTab === 'tasks') loadKnowledgeTasks();
            if (nextTab === 'jobs') loadProcessingJobs();
            if (nextTab === 'resignation') loadHandoverRecords();
            if (nextTab === 'feishu') loadFeishuWorkspace();
            if (nextTab === 'settings' && can('settings.manage')) {
                loadOperationsOverview();
            }
        });

        watch(chatMode, (nextMode) => {
            useRag.value = nextMode !== 'direct';
        });

        watch(useRag, (enabled) => {
            if (!enabled) chatMode.value = 'direct';
            else if (chatMode.value === 'direct') chatMode.value = 'smart';
        });

        watch(feishuView, (nextView) => {
            if (nextView === 'sources' && !feishuDiscoveredChats.value.length && !feishuDiscoveringChats.value) {
                discoverFeishuChats();
            }
            if (nextView === 'sources' && feishuSelectedGroup.value) {
                selectFeishuGroup(feishuSelectedGroup.value);
            }
            if (nextView === 'content') {
                loadFeishuContentData({ resetPage: true });
            }
        });

        watch([
            feishuContentSection,
            feishuHistoryType,
            feishuContentGroupId,
            feishuContentStatus,
            feishuMessageType,
            feishuDateRange,
            feishuHistoryQuery,
        ], () => {
            if (tab.value === 'feishu' && feishuView.value === 'content') {
                scheduleFeishuContentReload();
            }
        });

        watch([graphSearch, graphGroupFilter, graphLabelMode], () => {
            if (tab.value === 'graph') renderGraphSoon();
        });

        let ingestionAnimationTimer = null;
        let processingPollingTimer = null;

        async function initializeWorkspace() {
            await checkHealth();
            await loadDocuments();
            await loadStats();
            await loadGraph();
            await loadCategories();
            await loadGroupedDocs();
            await loadKnowledgeAssets();
            await loadHandoverRecords();
            await loadProjectMembers();
            await loadOnboardingTemplates();
            await loadOnboardingGuide();
            if (can('onboarding.manage')) await loadSupervisedOnboardingPlans();
            if (can('task.read')) await loadKnowledgeTasks({ keepSelection: false });
            if (can('task.read')) await loadKnowledgeTaskNotificationPolicy();
            if (can('job.read')) await loadProcessingJobs({ keepSelection: false, silent: true });
            if (can('feishu.read')) {
                await loadFeishuWorkspace();
                connectFeishuWorkspaceEvents();
            }
            if (can('settings.manage')) {
                await checkTunnelStatus();
                await loadOperationsOverview();
            }
            // 从 URL 参数读取 tab
            const urlParams = new URLSearchParams(window.location.search);
            const tabParam = urlParams.get('tab');
            if (tabParam && (!tabPermissions[tabParam] || can(tabPermissions[tabParam]))) tab.value = tabParam;
            if (tab.value === 'graph') renderGraphSoon();
            if (ingestionAnimationTimer) clearInterval(ingestionAnimationTimer);
            ingestionAnimationTimer = setInterval(() => {
                if (!kbUploading.value) activeIngestionStep.value = (activeIngestionStep.value + 1) % ingestionSteps.length;
            }, 2600);
            if (processingPollingTimer) clearInterval(processingPollingTimer);
            processingPollingTimer = setInterval(() => {
                if (tab.value === 'jobs' && (processingJobSummary.value.active || 0) > 0) {
                    loadProcessingJobs({ silent: true });
                }
            }, 2500);
        }

        onMounted(async () => {
            window.addEventListener('auth-required', handleAuthRequired);
            await loadAuthConfig();
            const authenticated = await loadCurrentUser();
            if (authenticated) await initializeWorkspace();
        });

        onBeforeUnmount(() => {
            window.removeEventListener('auth-required', handleAuthRequired);
            if (feishuEventSource) feishuEventSource.close();
            if (feishuContentReloadTimer) clearTimeout(feishuContentReloadTimer);
            if (ingestionAnimationTimer) clearInterval(ingestionAnimationTimer);
            if (processingPollingTimer) clearInterval(processingPollingTimer);
            handoverItemDraftTimers.forEach(timer => clearTimeout(timer));
            handoverItemDraftTimers.clear();
        });

        return {
            authReady, authConfig, currentUser, loginEmail, loginPassword, loginLoading, loginError,
            activationToken, activationPassword, activationConfirm, activationLoading, activationMessage,
            projectMembers, newMemberName, newMemberEmail, newMemberRole, newMemberDuty, memberSaving, memberMsg,
            latestActivationUrl, userInitial, can, login, activateAccount, logout,
            createProjectMember, updateProjectMember, copyActivationUrl,
            tab, mobileNavOpen, navigateTo, messages, input, loading, useRag, chatMode,
            documents, stats, messagesRef,
            currentTabLabel, pageTitle,
            activeRetrievalStep, retrievalSteps,
            activeIngestionStep, ingestionSteps,
            activeHandoverStep, handoverFlowSteps, handoverChecklist, handoverRisks,
            handoverStepClass, handoverStepStatusLabel,
            activeHandoverRecord, handoverRecords, acceptingHandoverId, handoverActionMessage,
            showHandoverModal, selectedHandover, handoverDetailLoading, handoverActionLoading,
            handoverItemDrafts, handoverItemAssetRefs, handoverItemAssetOptions, handoverItemAssetPickers,
            handoverItemDraftSaving, handoverItemDraftMessages,
            handoverRejectDrafts, handoverProxyReasonDrafts, handoverInventoryExcludeDrafts,
            handoverRiskActionDrafts, handoverRiskDraft,
            handoverItemStatusMeta, handoverRiskStatusMeta, handoverRiskSeverityMeta,
            handoverStatusLabel, handoverItemStatusLabel, handoverRiskStatusLabel, handoverMemberName,
            canSubmitHandoverItem, canReviewHandover, canReviewHandoverItem, canManageHandoverRisk,
            canRegisterHandoverRisk, isHandoverParticipant,
            handoverSectionOpen, handoverShowClosed, handoverInventoryExcludeActive, handoverShowRiskForm,
            handoverInventorySorted, handoverInventoryPending, handoverInventoryClosed,
            handoverItemsSorted, handoverItemsActionable, handoverItemsAccepted,
            handoverRisksSorted, handoverRisksActive, handoverRisksClosed,
            handoverMyActions, handoverSectionIsOpen, toggleHandoverSection, focusHandoverSection,
            handoverClosedVisible, toggleHandoverClosed,
            handoverInventoryExcludeIsActive, beginHandoverInventoryExclude, cancelHandoverInventoryExclude,
            handoverInventorySourceLabel, handoverInventoryStatusLabel,
            starterQuestions, dashboardDocs, graphPreviewNodes, graphPreviewEdges, feishuPreviewText,
            displayChunkCount, displayGraphNodes, displayGraphEdges, displayCitationCount,
            llmReady, storageReady, baseUrl,
            apiKey, apiKeyConfigured, apiBaseUrl, llmModel, llmProvider, providerLabel, savingKey, testingLLM, saveKeyMsg,
            llmTestMsg, llmTestOk, llmTestLatency, testLLMConnection,
            providerOptions, selectProvider, onProviderChange,
            proxyEnabled, proxyUrl, verifySsl, savingNetwork, networkMsg,
            resettingKnowledgeLibrary, knowledgeLibraryResetMessage, resetKnowledgeLibrary,
            governancePolicy, governancePreview, governanceBackups, governanceAudit, governanceAuditTotal,
            governanceAuditAction, governanceAuditKeyword, governanceConfirm, governanceLoading,
            governanceTarget, governanceTargetPreview, governanceSchedule,
            governanceActionLoading, governanceMessage, governanceAuditLabel, formatFileSize,
            governanceScheduleOperationLabel,
            operationsWindow, operationsOverview, operationsLoading, operationsActionLoading,
            operationsMessage, maintenanceDraft, operationsStatusLabel, operationsMetricValue,
            loadOperationsOverview, runCapacityValidation, createMaintenanceWindow,
            cancelMaintenanceWindow, runRecoveryDrill,
            feishuConfigured, feishuLoading, feishuMsg, pushChatId,
            feishuAppId, feishuAppSecret, feishuDefaultChatId,
            feishuWorkspace, feishuView, feishuGroups, feishuMessages, feishuCandidates, feishuAssets, feishuAudit,
            feishuContentAssets,
            feishuVisibleMessages, feishuContentGroups, feishuPendingMessages, feishuPendingCandidates, feishuAllPendingSelected,
            feishuPublishedAssets, feishuAllAssetsSelected, feishuSummary, feishuConnection,
            feishuArchiveCount, feishuHistoryCount,
            feishuTransport, feishuDiagnostics, feishuEventConnected,
            feishuPipelineStages, feishuTodayConversion, feishuProcessingComposition, feishuRecentActivity,
            feishuLiveSyncState,
            feishuDiscoveredChats, feishuAvailableChats, feishuProjectGroups, feishuDiscoveringChats, feishuDiscoveryMsg,
            feishuSelectedGroupId, feishuSelectedGroup, feishuContentGroupId, feishuHistoryQuery, feishuContentStatus, feishuMessageType,
            feishuContentSection, feishuHistoryType, feishuDateRange, feishuContentLoading,
            feishuContentPage, feishuContentTotal, feishuContentPages,
            feishuWorkspaceLoading, feishuSyncing, feishuSavingGroup, feishuReviewingId, feishuEnrichingId,
            feishuShowCredentialForm, feishuGroupDraft, feishuCategoryOptions,
            feishuSelectedCandidateIds, feishuSelectedAssetIds, feishuBatchCategory, feishuBatchLoading,
            feishuRevokeConfirmOpen, feishuRevokeReason, feishuRevokeTargets,
            feishuPolicyPreview, feishuPolicyPreviewLoading, feishuPermissionResult, feishuPermissionLoading, feishuUndoAsset,
            // 知识库
            categories, showUploadModal, showCatModal,
            uploaderName, uploadCategory, uploadFileSelected, uploadFileName, uploadFileData, uploadFilesData,
            uploadMode, uploadTargetAssetId, uploadTargetAsset, uploadUpdatableAssets, uploadInputKey,
            kbUploading, kbUploadMsg, knowledgeTaskUploadContext, lastIngestion, graphImportHint, groupedDocs, kbLoading, kbLoadError, previewData, previewFilename,
            knowledgeAssets, knowledgeAssetSummary, knowledgeAssetsLoading, knowledgeAssetsError, knowledgeAssetsAccessDenied,
            knowledgeSearch, knowledgeStatusFilter, knowledgeCategoryFilter, knowledgeStatusOptions, knowledgeActionMetrics, filteredKnowledgeAssets,
            selectedKnowledgeAsset, knowledgeAssetVersions, knowledgeAssetDetailLoading, knowledgeAssetDetailError, knowledgeAssetDetailAccessDenied,
            knowledgeAssetDetailUnavailable, knowledgeAssetSaving, knowledgeAssetMessage, knowledgeAssetDraft, knowledgeRevokeReason,
            knowledgeRevokeConfirmOpen, knowledgeVersionPreviewingId,
            newCatName, editingCatIdx, editingCatValue,
            loadCategories, addCategory, startCatEdit, saveCatEdit, deleteCategory,
            openUploadModal, closeUploadModal, setUploadMode, onKbFileChange, submitKbUpload, loadGroupedDocs, previewDoc, categoryLabel, isBuiltInCategory,
            loadKnowledgeAssets, openKnowledgeAsset, closeKnowledgeAsset, previewKnowledgeAssetSource,
            saveKnowledgeAsset, transitionKnowledgeAsset, knowledgeAvailableTransitions, setKnowledgeStatusFilter,
            knowledgeAssetNeedsRepair, repairKnowledgeAssetProjections,
            knowledgeAssetId, knowledgeAssetTitle, knowledgeAssetSource, knowledgeAssetStatus, knowledgeStatusLabel, knowledgeStatusTone,
            knowledgeOwnerName, knowledgeVersionLabel, knowledgeReviewMeta, knowledgeVisibilityLabel, knowledgeVisibilityDetail,
            knowledgeVisibilityOptions, knowledgeSourceVisibility,
            knowledgeAssetCategoryLabel, knowledgeProcessingStatus, knowledgeCurrentVersion, knowledgeVersionDocuments,
            knowledgeProjectionItems, knowledgeProjectionLabel, knowledgeLatestReadyVersion,
            knowledgeVersionName, knowledgeVersionStatus, isKnowledgeCurrentVersion, previewKnowledgeVersion, knowledgeMessageIsError, formatKnowledgeDate,
            knowledgeAssetOptionLabel,
            // 对话
            renderMarkdown, scrollToBottom, toggleThinking, formatProcessDuration, processStatusLabel,
            sendMessage, clearChat, submitAnswerFeedback,
            ragEvaluationEnabled, ragEvaluationOpen, ragEvaluationSets, selectedRagEvaluationSet, ragEvaluationLoading,
            ragEvaluationMessage, ragEvaluationSetDraft, ragEvaluationCaseDraft,
            openRagEvaluation, selectRagEvaluationSet, createRagEvaluationSet,
            addRagEvaluationCase, runRagEvaluation,
            // 文档
            loadDocuments, deleteDoc, loadStats,
            // 图谱
            graphData, graphError, graphBuilding, graphCanvasRef,
            graphSearch, graphGroupFilter, graphLabelMode, graphGroups, graphLegendGroups, filteredGraph, graphTableNodes, graphTableEdges, graphZoomLabel,
            graphSaving, graphMsg, selectedGraphNode, selectedGraphEdge, graphEditorMode,
            nodeDraft, newNodeDraft, edgeDraft,
            loadGraph, buildGraph, selectGraphNode, selectGraphEdge, clearGraphSelection,
            applyNodeDraft, addGraphNode, deleteGraphNode,
            applyEdgeDraft, addGraphEdge, deleteGraphEdge, resetGraphView,
            // 飞书
            pushDaily, saveFeishuSettings, loadFeishuWorkspace, startFeishuEventConnection, selectFeishuGroup, startFeishuGroupDraft, saveFeishuGroup, syncFeishuMessages,
            reviewFeishuMessage, previewFeishuPolicy, checkFeishuPermissions,
            feishuResourceUrls, feishuVisibleLinks, feishuDisplayContent, feishuExtractionLabel, retryFeishuEnrichment,
            toggleFeishuCandidate, toggleAllFeishuCandidates, batchFeishuCandidates, revertFeishuAsset,
            toggleFeishuAsset, toggleAllFeishuAssets, batchRevertFeishuAssets,
            closeFeishuRevokeConfirm, confirmRevertFeishuAssets, feishuMessageTone,
            feishuModeLabel, feishuReviewLabel, feishuMessageTypeLabel, feishuContentStatusLabel, feishuCandidateStatusLabel, feishuAuditLabel, formatFeishuTime,
            feishuTransportLabel, feishuActionLabel,
            feishuLiveSyncLabel,
            discoverFeishuChats, selectFeishuAvailableChat, openFeishuView,
            loadFeishuContentData, selectFeishuContentSection, selectFeishuHistoryType, changeFeishuContentPage,
            // 离职交接
            resignName, resignRecipient, resignDueDate, resignFiles,
            resignSubmitting, resignResult, resignError, resignResultNeedsRepair, resignRepairDetails, handoverLoading, handoverLoadError,
            onResignFilesChange, submitResignation, loadHandoverRecords, acceptHandover, cancelHandover, canAcceptHandover, canCancelHandover, openHandoverLearningPlan,
            openHandoverDetail, actOnHandoverItem, refreshHandoverInventory,
            handoverItemHasDraftEvidence, handoverEvidenceReferenceLabel,
            loadHandoverItemSubmissionContext, scheduleHandoverItemDraftSave,
            saveHandoverItemDraft, addHandoverItemExistingAsset, removeHandoverItemDraftAsset,
            uploadHandoverItemEvidenceFiles,
            actOnHandoverInventory, createHandoverRisk, actOnHandoverRisk, completeHandover, cancellingHandoverId,
            // 新人赋能
            onboardingRole, onboardingRoles, onboardingRoleLabel, onboardingTemplates, dutyOptions,
            onboardingTargetUserId, onboardingTargetMembers, onboardingData, onboardingLoading,
            onboardingError, onboardingActionLoading, onboardingEvidenceDrafts,
            supervisedOnboardingPlans, supervisedOnboardingPlansLoading, supervisedOnboardingPlansError,
            supervisedOnboardingSummary, onboardingPlanDetailOpen, onboardingPlanDetailId,
            showOnboardingTemplateModal, onboardingTemplateDraft,
            loadOnboardingTemplates, loadOnboardingGuide, createOnboardingPlan,
            updateOnboardingItem, refreshOnboardingPlanMaterials,
            loadSupervisedOnboardingPlans, openOnboardingPlanAssignment, openSupervisedOnboardingPlan,
            closeOnboardingPlanDetail, confirmSupervisedOnboardingPlan,
            onboardingItemTypeLabel, onboardingItemStatusLabel, readinessDimensionLabel,
            openOnboardingTemplateEditor, addOnboardingTopic, removeOnboardingTopic, saveOnboardingTemplate,
            onboardingTaskMessage, createOnboardingGapTasks,
            // 知识任务
            knowledgeTasks, knowledgeTaskSummary, knowledgeTasksLoading, knowledgeTasksError,
            selectedKnowledgeTask, knowledgeTaskDetailLoading, knowledgeTaskStatusFilter,
            knowledgeTaskTypeFilter, knowledgeTaskKeyword, knowledgeTaskMessage, knowledgeTaskActionLoading,
            knowledgeTaskEvidence, knowledgeTaskEvidenceUploadMessage, openKnowledgeTaskUploadModal,
            knowledgeTaskNote, knowledgeTaskAssignee, knowledgeTaskChatId,
            showKnowledgeTaskCancelModal, knowledgeTaskCancelReason, knowledgeTaskCancelMessage,
            showKnowledgeTaskModal, knowledgeTaskSaving, knowledgeTaskDraft,
            showKnowledgeTaskPolicyModal, knowledgeTaskPolicyLoading, knowledgeTaskPolicySaving,
            knowledgeTaskPolicyScanning, knowledgeTaskPolicyMessage, knowledgeTaskNotificationPolicy,
            knowledgeTaskStatusMeta, knowledgeTaskTypeMeta, knowledgeTaskCreatableTypeMeta, knowledgeTaskPriorityMeta,
            canManageKnowledgeTasks, filteredKnowledgeTasks, knowledgeTaskPolicyStatus,
            knowledgeTaskCan,
            knowledgeTaskStatusLabel, knowledgeTaskTypeLabel, knowledgeTaskPriorityLabel,
            knowledgeTaskEventLabel, knowledgeTaskNotificationKindLabel, knowledgeTaskNotificationStatusLabel,
            knowledgeTaskWritebackTitle, knowledgeTaskWritebackStatus,
            openKnowledgeTaskProcessingJob,
            openKnowledgeTaskModal, loadKnowledgeTasks, openKnowledgeTask, createKnowledgeTask,
            actOnKnowledgeTask, openKnowledgeTaskCancel, closeKnowledgeTaskCancel, confirmKnowledgeTaskCancel,
            notifyKnowledgeTask, openFeishuReviewTask, openHandoverGapTask,
            openKnowledgeTaskPolicy, loadKnowledgeTaskNotificationPolicy,
            saveKnowledgeTaskNotificationPolicy, scanKnowledgeTaskNotifications,
            // 处理任务
            processingJobs, processingJobSummary, processingJobsLoading, processingJobsError,
            selectedProcessingJob, processingJobDetailLoading, processingJobStatusFilter,
            processingJobTypeFilter, processingJobKeyword, processingJobPage, processingJobPages,
            processingJobTotal, processingJobActionLoading, processingJobMessage,
            processingJobStatusMeta, processingJobTypeMeta, canManageProcessingJobs,
            canRetrySelectedProcessingJob,
            processingJobStatusLabel, processingJobTypeLabel, processingJobStageLabel,
            processingJobDisplayName,
            loadProcessingJobs, openProcessingJob, actOnProcessingJob, changeProcessingJobPage,
            // 设置
            saveLLMSettings, saveNetworkSettings, restartServer,
            loadDataGovernance, loadGovernanceAudit, saveDataGovernancePolicy,
            createDataBackup, applyDataRetention, exportGovernanceAudit,
            scanDataGovernanceSchedule,
            previewGovernanceTarget, exportGovernanceTarget, deleteGovernanceTarget,
            // Tunnel
            tunnelRunning, tunnelUrl, tunnelLoading, tunnelMsg,
            startTunnel, stopTunnel, checkTunnelStatus,
        };
    }
}).mount('#app');
