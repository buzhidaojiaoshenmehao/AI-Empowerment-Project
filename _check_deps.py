"""依赖完整性与关键版本检查。"""
import sys, traceback
from importlib.metadata import PackageNotFoundError, version

sys.path.insert(0, '.')
ok = True

if sys.version_info < (3, 10):
    print(f'  [FAIL] Python >= 3.10 required, current: {sys.version.split()[0]}')
    ok = False
else:
    print(f'  [OK] Python {sys.version.split()[0]}')

def check(mod_name, pip_name):
    global ok
    try:
        __import__(mod_name)
        print(f'  [OK] {pip_name}')
    except ImportError as e:
        print(f'  [FAIL] {pip_name}: {e}')
        ok = False

check('fastapi', 'fastapi')
check('uvicorn', 'uvicorn')
check('langchain_core.documents', 'langchain-core')
check('langchain.agents', 'langchain')
check('langchain_text_splitters', 'langchain-text-splitters')
check('langchain_openai', 'langchain-openai')
check('langgraph', 'langgraph')
check('numpy', 'numpy')
check('pypdf', 'pypdf')
check('docx', 'python-docx')
check('docx2txt', 'docx2txt')
check('pydantic_settings', 'pydantic-settings')
check('multipart', 'python-multipart')
check('httpx', 'httpx')

required_versions = {
    'langchain': '1.3.15',
    'langchain-core': '1.5.4',
    'langchain-openai': '1.4.3',
    'langchain-text-splitters': '1.1.2',
    'langgraph': '1.2.11',
}
for package, expected in required_versions.items():
    try:
        installed = version(package)
        if installed != expected:
            print(f'  [FAIL] {package}=={expected} required, current: {installed}')
            ok = False
        else:
            print(f'  [OK] {package}=={installed}')
    except PackageNotFoundError:
        print(f'  [FAIL] {package} is not installed')
        ok = False

try:
    from langchain.agents import create_agent
    assert callable(create_agent)
    print('  [OK] LangChain create_agent API')
except (ImportError, AssertionError) as exc:
    print(f'  [FAIL] LangChain create_agent API: {exc}')
    ok = False

print()
print('后端模块加载...')
try:
    from backend.knowledge_base.vector_store import vector_store
    from backend.knowledge_base.llm_service import llm_service
    from backend.knowledge_base.document_loader import load_document
    from backend.knowledge_base.retriever import retriever
    from backend.knowledge_base.project_context import project_context
    from backend.knowledge_base.knowledge_graph import knowledge_graph
    print('  [OK] 所有后端模块加载成功')
except Exception as e:
    print(f'  [FAIL] {e}')
    traceback.print_exc()
    ok = False

print()
print('ALL_CHECKS_PASSED' if ok else 'SOME_CHECKS_FAILED')
raise SystemExit(0 if ok else 1)
