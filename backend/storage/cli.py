"""Offline SQLite maintenance commands.

Examples:
    python -m backend.storage.cli status
    python -m backend.storage.cli quick-check
    python -m backend.storage.cli backup
    python -m backend.storage.cli restore data/backups/app-....db
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from backend.storage import backup_dir, database_path, get_repository, initialize_storage
from backend.storage.database import Database


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI 赋能项目 SQLite 维护工具")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="查看 schema 与旧数据迁移状态")
    subparsers.add_parser("quick-check", help="执行 SQLite quick_check 与外键检查")
    backup = subparsers.add_parser("backup", help="在线一致性备份数据库")
    backup.add_argument("--name", default="", help="可选备份文件名")
    restore = subparsers.add_parser("restore", help="从备份恢复，恢复前自动安全备份")
    restore.add_argument("path", type=Path, help="备份数据库路径")
    member = subparsers.add_parser("provision-member", help="创建项目成员或更新其角色")
    member.add_argument("--email", required=True)
    member.add_argument("--name", required=True)
    member.add_argument("--role", default="project_member")
    invite = subparsers.add_parser("invite-member", help="生成一次性账号激活信息")
    invite.add_argument("--email", required=True)
    invite.add_argument("--output", type=Path, required=True, help="写入权限为 600 的本地文件")
    subparsers.add_parser("list-members", help="列出当前项目成员（不含凭据）")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "restore":
        try:
            result = Database(database_path()).restore(args.path, backup_dir())
        except (FileNotFoundError, RuntimeError) as exc:
            print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False, indent=2))
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    # Status, integrity checks and online backup may run beside the app. Only
    # restore claims the exclusive runtime lock and therefore requires downtime.
    initialize_storage(run_legacy_migration=False, enforce_single_process=False)
    database = get_repository().database
    if args.command == "status":
        result = database.migration_status()
    elif args.command == "quick-check":
        result = database.quick_check()
    elif args.command == "backup":
        result = database.backup(backup_dir(), name=args.name)
    elif args.command == "provision-member":
        from backend.auth import ROLE_PERMISSIONS

        if args.role not in ROLE_PERMISSIONS:
            print(json.dumps({"success": False, "error": "不支持的项目角色"}, ensure_ascii=False, indent=2))
            return 2
        result = {"success": True, "member": get_repository().provision_project_member(
            args.email, args.name, args.role,
        )}
        result["member"].pop("password_hash", None)
    elif args.command == "invite-member":
        from backend.auth import auth_service

        user = get_repository().get_user_by_email(args.email)
        if not user:
            print(json.dumps({"success": False, "error": "用户不存在"}, ensure_ascii=False, indent=2))
            return 2
        invitation = auth_service.create_invitation(str(user["user_id"]))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            **invitation,
            "activation_path": f"/?activate={invitation['token']}",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        os.chmod(args.output, 0o600)
        result = {"success": True, "output": str(args.output.resolve()), "expires_at": invitation["expires_at"]}
    elif args.command == "list-members":
        result = {"success": True, "members": get_repository().list_project_members()}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
