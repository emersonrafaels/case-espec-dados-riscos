"""AWS Athena client for querying the case database.

## Autenticação suportada

1. **Chaves explícitas** — preencha `aws_access_key_id` + `aws_secret_access_key`
   (+ `aws_session_token` para credenciais temporárias / STS).

2. **AWS SSO** — deixe as chaves em branco e informe `profile_name`:
       $ aws sso login --profile <profile_name>
   O boto3 lê as credenciais SSO automaticamente a partir do perfil configurado
   em ~/.aws/config.  Se o perfil padrão já tiver SSO, `profile_name` pode
   ser omitido e o boto3 ainda usará a cadeia de credenciais padrão.

3. **Role / instance profile / env vars** — deixe tudo em branco; o boto3
   percorre a cadeia padrão: env vars → ~/.aws/credentials → IMDSv2.

Database e tabela são injetados via construtor (sem constantes hardcoded).
Ver src/config/settings.toml  [athena]  para os valores padrão.
"""

import time
from typing import Optional

import boto3
import pandas as pd


class AthenaClient:
    def __init__(
        self,
        region: str,
        database: str,
        table: str,
        s3_output: str = "",
        workgroup: str = "primary",
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        profile_name: Optional[str] = None,
    ) -> None:
        self.database  = database
        self.table     = table
        self.workgroup = workgroup
        self.s3_output = s3_output.rstrip("/") if s3_output else ""

        session_kwargs: dict = {"region_name": region}
        if profile_name:
            session_kwargs["profile_name"] = profile_name

        session = boto3.Session(**session_kwargs)

        # Somente passa chaves explícitas se ambas estiverem presentes;
        # caso contrário o boto3 usa a cadeia de credenciais da Session
        # (inclui SSO, instance profile, env vars, etc.).
        self._client_kwargs: dict = {}
        if aws_access_key and aws_secret_key:
            self._client_kwargs["aws_access_key_id"]     = aws_access_key
            self._client_kwargs["aws_secret_access_key"] = aws_secret_key
            if aws_session_token:
                self._client_kwargs["aws_session_token"] = aws_session_token

        self._athena = session.client("athena", **self._client_kwargs)
        self._glue   = session.client("glue",   **self._client_kwargs)

    # ── Public API ─────────────────────────────────────────────────────────────

    def list_tables(self, database: Optional[str] = None) -> list:
        """Retorna os nomes de todas as tabelas do database via Glue Catalog."""
        db     = database or self.database
        tables = []
        kwargs = {"DatabaseName": db}
        while True:
            resp = self._glue.get_tables(**kwargs)
            tables.extend(t["Name"] for t in resp.get("TableList", []))
            token = resp.get("NextToken")
            if not token:
                break
            kwargs["NextToken"] = token
        return sorted(tables)

    def query(
        self,
        sql: str,
        database: Optional[str] = None,
        timeout_seconds: int = 120,
    ) -> pd.DataFrame:
        """Execute *sql* e retorna os resultados como DataFrame."""
        db = database or self.database
        execution_id = self._start(sql, db)
        self._wait(execution_id, timeout_seconds)
        return self._fetch(execution_id)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _start(self, sql: str, database: str) -> str:
        kwargs: dict = {
            "QueryString": sql,
            "QueryExecutionContext": {"Database": database},
            "WorkGroup": self.workgroup,
        }
        # ResultConfiguration é opcional quando o workgroup já define o output S3.
        if self.s3_output:
            kwargs["ResultConfiguration"] = {"OutputLocation": self.s3_output}
        resp = self._athena.start_query_execution(**kwargs)
        return resp["QueryExecutionId"]

    def _wait(self, execution_id: str, timeout_seconds: int) -> None:
        terminal = {"SUCCEEDED", "FAILED", "CANCELLED"}
        for _ in range(timeout_seconds):
            resp  = self._athena.get_query_execution(QueryExecutionId=execution_id)
            state = resp["QueryExecution"]["Status"]["State"]
            if state == "SUCCEEDED":
                return
            if state in terminal:
                reason = resp["QueryExecution"]["Status"].get("StateChangeReason", "—")
                raise RuntimeError(f"Query {state}: {reason}")
            time.sleep(1)
        raise TimeoutError(
            f"Athena query timed out after {timeout_seconds}s (id={execution_id})"
        )

    def _fetch(self, execution_id: str) -> pd.DataFrame:
        paginator = self._athena.get_paginator("get_query_results")
        headers: Optional[list] = None
        rows: list = []

        for page in paginator.paginate(QueryExecutionId=execution_id):
            page_rows = page["ResultSet"]["Rows"]
            if headers is None and page_rows:
                headers   = [col.get("VarCharValue", "") for col in page_rows[0]["Data"]]
                page_rows = page_rows[1:]
            rows.extend(
                [col.get("VarCharValue", "") for col in row["Data"]]
                for row in page_rows
            )

        if not headers:
            return pd.DataFrame()
        return pd.DataFrame(rows, columns=headers)
