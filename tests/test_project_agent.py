import json
import unittest

from backend.knowledge_base.project_agent import ProjectAgentService, ProjectAgentTraceCallback


class ProjectAgentTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.service = ProjectAgentService(
            onboarding_provider=lambda role: {
                "role": role,
                "readiness": 50,
                "focus": "先掌握项目资料",
                "gaps": [{"label": "风险日志"}],
                "learning_path": ["阅读资料"],
                "recommended_documents": [{"source_file": "新人指南.md", "category_label": "新人培训"}],
            },
            handover_provider=lambda: [{
                "id": "handover-1",
                "name": "张三",
                "role": "开发工程师",
                "recipient": "李四",
                "status": "pending_acceptance",
                "completion": 75,
                "file_count": 2,
                "risks": ["接替人尚未确认接收"],
            }],
        )

    def test_agent_exposes_only_read_tools(self):
        tool_names = {tool.name for tool in self.service.build_tools()}
        self.assertEqual(tool_names, {
            "search_project_knowledge",
            "trace_knowledge_graph",
            "get_onboarding_guide",
            "get_handover_status",
            "get_project_overview",
        })
        self.assertFalse(any(word in name for name in tool_names for word in ("delete", "upload", "save", "push", "approve")))

    def test_onboarding_and_handover_tools_return_structured_results(self):
        tools = {tool.name: tool for tool in self.service.build_tools()}
        onboarding = json.loads(tools["get_onboarding_guide"].invoke({"role": "测试工程师"}))
        handover = json.loads(tools["get_handover_status"].invoke({"query": "张三"}))
        self.assertEqual(onboarding["readiness"], 50)
        self.assertEqual(onboarding["gap_count"], 1)
        self.assertEqual(handover["record_count"], 1)
        self.assertEqual(handover["risk_count"], 1)

    async def test_tool_callback_emits_real_progress_and_sources(self):
        callback = ProjectAgentTraceCallback()
        run_id = "run-1"
        await callback.on_tool_start({"name": "search_project_knowledge"}, "SRP", run_id=run_id)
        await callback.on_tool_end(
            json.dumps({
                "result_count": 1,
                "sources": [{"source": "导航设计.md", "content": "SRP 功能说明"}],
            }, ensure_ascii=False),
            run_id=run_id,
        )
        started = await callback.queue.get()
        finished = await callback.queue.get()
        sources = await callback.queue.get()
        self.assertEqual(started["payload"]["status"], "active")
        self.assertEqual(finished["payload"]["status"], "done")
        self.assertEqual(sources["sources"][0]["source"], "导航设计.md")

    async def test_project_overview_cannot_mask_empty_query_evidence(self):
        callback = ProjectAgentTraceCallback()
        callback.tool_runs["search-run"] = "search_project_knowledge"
        await callback.on_tool_end(
            json.dumps({"result_count": 0, "sources": []}, ensure_ascii=False),
            run_id="search-run",
        )
        callback.tool_runs["overview-run"] = "get_project_overview"
        await callback.on_tool_end(
            json.dumps({"document_count": 7, "sources": []}, ensure_ascii=False),
            run_id="overview-run",
        )

        self.assertTrue(callback.query_evidence_checked)
        self.assertFalse(callback.query_evidence_found)


if __name__ == "__main__":
    unittest.main()
