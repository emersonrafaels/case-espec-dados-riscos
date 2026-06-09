"""Chat interface — Obsolescência de Servidores via IARA GenAI + AWS Athena.

Run:
    streamlit run src/chat/app.py
"""

# ── Namespace shim (must run before any iara import) ──────────────────────────
import sys
from pathlib import Path

_CHAT_DIR = Path(__file__).parent
_SRC_DIR  = _CHAT_DIR.parent

sys.path.insert(0, str(_CHAT_DIR))   # iara_setup, athena_client
sys.path.insert(0, str(_SRC_DIR))    # config package (src/config/)

import iara_setup

iara_setup.setup()
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import streamlit as st

from athena_client import AthenaClient
from config import get_config
from construct_cost_ai.infra.ai.frameworks.iara.src.agents.chat import IaraAgentChat
from construct_cost_ai.infra.ai.frameworks.iara.src.config.iara_config import get_iara_config
from construct_cost_ai.infra.ai.frameworks.iara.src.models.llm import IaraLLMConfig

# ── Config ────────────────────────────────────────────────────────────────────
_cfg = get_config()
_ia  = _cfg["ia"]
_ath = _cfg["athena"]
_app = _cfg["app"]

_PROVIDERS = ["openai", "azure_openai", "bedrock", "vertex"]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=_app["title"],
    page_icon=_app["icon"],
    layout=_app["layout"],
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _default_sql(
    table: str,
    limit: int | None = None,
    partition_col: str = "",
    partitions: list | None = None,
) -> str:
    """Gera o SQL padrão aplicando filtro de partição e LIMIT quando informados."""
    where = ""
    if partition_col and partitions:
        if len(partitions) == 1:
            where = f" WHERE {partition_col} = '{partitions[0]}'"
        else:
            vals  = ", ".join(f"'{p}'" for p in partitions)
            where = f" WHERE {partition_col} IN ({vals})"
    limit_clause = f" LIMIT {limit}" if limit else ""
    return f'SELECT * FROM "{_ath["database"]}"."{table}"{where}{limit_clause};'


def _build_system_prompt(df: pd.DataFrame, table: str, is_sample: bool = True) -> str:
    cols     = "\n".join(f"  - `{c}`" for c in df.columns)
    table_md = df.to_markdown(index=False) if not df.empty else "_(vazio)_"
    n        = len(df)

    if is_sample:
        data_header    = f"## Dados carregados (amostra — {n} registros)"
        data_directive = (
            "- Os dados acima são uma **amostra** da tabela. "
            "Se a pergunta exigir análise do dataset completo, informe claramente "
            "e sugira uma query SQL específica para obtê-los no Athena."
        )
    else:
        data_header    = f"## Dados carregados — dataset completo ({n} registros)"
        data_directive = (
            "- Os dados acima representam o **dataset completo** da tabela. "
            "Responda diretamente com base neles. "
            "**Não sugira queries SQL para obter mais dados** — eles já estão todos carregados."
        )

    return f"""Você é um assistente analítico especializado em **obsolescência de servidores** do Itaú Unibanco.

Você tem acesso a dados da tabela `{table}` no AWS Athena \
(banco: `{_ath["database"]}`).

## Colunas disponíveis
{cols}

{data_header}
{table_md}

**Diretrizes:**
- Responda sempre em **português brasileiro**.
- Baseie suas análises nos dados acima; não invente valores.
- Use tabelas markdown e números formatados quando útil.
{data_directive}
- Seja objetivo, analítico e preciso.
"""


def _make_agent(
    provider: str, api_key: str, client_id: str, client_secret: str,
    model: str, temperature: float, system_prompt: str = "",
) -> IaraAgentChat:
    if hasattr(get_iara_config, "cache_clear"):
        get_iara_config.cache_clear()
    return IaraAgentChat(
        provider=provider,
        api_key=api_key or None,
        client_id=client_id or None,
        client_secret=client_secret or None,
        llm_config=IaraLLMConfig(model=model, temperature=temperature),
        system_prompt=system_prompt or None,
    )


def _make_athena(region, s3_output, table, workgroup, aws_key, aws_secret, aws_token, aws_profile) -> AthenaClient:
    return AthenaClient(
        region=region,
        database=_ath["database"],
        table=table,
        s3_output=s3_output,
        workgroup=workgroup or "primary",
        aws_access_key=aws_key or None,
        aws_secret_key=aws_secret or None,
        aws_session_token=aws_token or None,
        profile_name=aws_profile or None,
    )


# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "messages":            [],
    "context":             [],
    "df":                  pd.DataFrame(),
    "agent":               None,
    "available_tables":    [],
    "selected_table":      _ath["table"],
    "available_partitions": [],
    "selected_partitions": [_ath["default_partition"]] if _ath.get("default_partition") else [],
    "data_is_sample":      True,
}
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configurações")

    # ── Agente de IA ───────────────────────────────────────────────────────────
    with st.expander("🤖 Agente de IA", expanded=True):
        provider_idx = _PROVIDERS.index(_ia["provider"]) if _ia["provider"] in _PROVIDERS else 0
        provider = st.selectbox("Provider", _PROVIDERS, index=provider_idx,
            help="Selecione **openai** para uso externo com API key. "
                 "Os demais requerem credenciais internas Itaú (iaragenai SDK).")

        if provider == "openai":
            api_key       = st.text_input("OpenAI API Key", value=_ia["api_key"], type="password", placeholder="sk-...")
            client_id     = client_secret = ""
        else:
            api_key       = ""
            client_id     = st.text_input("Client ID",     value=_ia["client_id"],     type="password")
            client_secret = st.text_input("Client Secret", value=_ia["client_secret"], type="password")

        model       = st.text_input("Modelo", value=_ia["model"])
        temperature = st.slider("Temperatura", 0.0, 1.0, float(_ia["temperature"]), 0.05)

        if st.button("🔌 Conectar IA", use_container_width=True, type="primary"):
            try:
                with st.spinner("Inicializando agente..."):
                    sys_prompt = (
                        _build_system_prompt(
                            st.session_state.df,
                            st.session_state.selected_table,
                            st.session_state.data_is_sample,
                        )
                        if not st.session_state.df.empty else ""
                    )
                    st.session_state.agent = _make_agent(
                        provider, api_key, client_id, client_secret, model, temperature, sys_prompt)
                st.success("Agente conectado!")
            except Exception as exc:
                st.error(f"Erro ao conectar: {exc}")

        if st.session_state.agent:
            st.success("✅ IA conectada")
        else:
            st.info("ℹ️ Clique em **Conectar IA**.")

    # ── Athena ─────────────────────────────────────────────────────────────────
    with st.expander("☁️ AWS Athena", expanded=True):
        aws_region  = st.text_input("Região",    value=_ath["region"])
        workgroup   = st.text_input("Workgroup", value=_ath["workgroup"],
                          help="Workgroup do Athena (padrão: primary). env: ATHENA_WORKGROUP")
        s3_output   = st.text_input("S3 Output", value=_ath["s3_output"],
                          placeholder="s3://meu-bucket/athena-results/",
                          help="Opcional quando o workgroup já define o output S3.")
        aws_key     = st.text_input("Access Key ID",     value=_ath["aws_access_key_id"],     type="password")
        aws_secret  = st.text_input("Secret Access Key", value=_ath["aws_secret_access_key"], type="password")
        aws_token   = st.text_input("Session Token",     value=_ath["aws_session_token"],     type="password",
                          placeholder="opcional — credenciais temporárias")
        aws_profile = st.text_input("AWS Profile", value=_ath["profile"], placeholder="ex: case-riscos",
                          help="**AWS SSO:** deixe as chaves em branco, informe o perfil e rode:\n"
                               "`aws sso login --profile <perfil>`")

        st.divider()

        # ── Tabela ─────────────────────────────────────────────────────────────
        if st.button("🔍 Listar Tabelas", use_container_width=True):
            try:
                with st.spinner("Buscando tabelas..."):
                    tmp = _make_athena(aws_region, s3_output.strip(), _ath["table"],
                                       workgroup, aws_key, aws_secret, aws_token, aws_profile)
                    st.session_state.available_tables    = tmp.list_tables()
                    st.session_state.available_partitions = []   # reset ao trocar de tabela
                st.success(f"{len(st.session_state.available_tables)} tabelas encontradas.")
            except Exception as exc:
                st.error(f"Erro ao listar tabelas: {exc}")

        table_options = st.session_state.available_tables or [_ath["table"]]
        current_table = st.session_state.selected_table
        default_idx   = table_options.index(current_table) if current_table in table_options else 0
        selected_table = st.selectbox("Tabela", table_options, index=default_idx,
                             help="Clique em **Listar Tabelas** para carregar todas as tabelas.")
        if selected_table != st.session_state.selected_table:
            st.session_state.available_partitions  = []   # reset ao trocar de tabela
            st.session_state.selected_partitions   = [_ath["default_partition"]]
        st.session_state.selected_table = selected_table

        st.divider()

        # ── Partições ──────────────────────────────────────────────────────────
        partition_col = st.text_input("Coluna de partição", value=_ath["partition_col"],
                            help="Nome da coluna de partição. env: ATHENA_PARTITION_COL")

        if st.button("🔍 Listar Partições", use_container_width=True):
            try:
                with st.spinner("Buscando partições..."):
                    tmp = _make_athena(aws_region, s3_output.strip(), selected_table,
                                       workgroup, aws_key, aws_secret, aws_token, aws_profile)
                    st.session_state.available_partitions = tmp.list_partitions()

                found = len(st.session_state.available_partitions)
                if found == 0 and _ath.get("default_partition"):
                    st.success("1 partição encontrada (valor padrão do settings.toml).")
                else:
                    st.success(f"{found} partição(ões) encontrada(s).")
            except Exception as exc:
                st.error(f"Erro ao listar partições: {exc}")

        part_options = st.session_state.available_partitions or [_ath["default_partition"]]
        # Garante que os valores selecionados existam nas opções
        prev_selected = [p for p in st.session_state.selected_partitions if p in part_options]
        if not prev_selected and _ath["default_partition"] in part_options:
            prev_selected = [_ath["default_partition"]]

        selected_partitions = st.multiselect(
            "Partições",
            options=part_options,
            default=prev_selected,
            help="Clique em **Listar Partições** para carregar os valores disponíveis. "
                 "Múltiplas seleções geram cláusula IN.",
        )
        st.session_state.selected_partitions = selected_partitions

        st.divider()

        # ── Limite de linhas ───────────────────────────────────────────────────
        no_limit = st.checkbox("Sem limite (retornar todas as linhas)")
        if no_limit:
            row_limit = None
        else:
            row_limit = st.slider(
                "Limite de linhas",
                min_value=_ath["limit_min"],
                max_value=_ath["limit_max"],
                value=min(_ath["limit_default"], _ath["limit_max"]),
                step=_ath["limit_min"],
            )

        st.divider()

        # ── SQL e carga ────────────────────────────────────────────────────────
        auto_sql = _default_sql(selected_table, row_limit, partition_col, selected_partitions or None)
        custom_sql = st.text_area(
            "SQL customizado (opcional)",
            placeholder=auto_sql,
            height=90,
            help="Deixe em branco para usar o SQL gerado automaticamente com os filtros acima.",
        )

        if st.button("📊 Carregar Dados", use_container_width=True, type="primary"):
            try:
                with st.spinner("Consultando Athena..."):
                    athena = _make_athena(aws_region, s3_output.strip(), selected_table,
                                          workgroup, aws_key, aws_secret, aws_token, aws_profile)
                    sql = custom_sql.strip() or auto_sql

                    try:
                        df = athena.query(sql)
                    except RuntimeError as exc:
                        # Se a coluna de partição não existir na tabela, reprocessa sem o filtro
                        _err = str(exc).lower()
                        _using_partition = (
                            not custom_sql.strip()
                            and partition_col
                            and selected_partitions
                            and (
                                "column" in _err
                                or "cannot be resolved" in _err
                                or partition_col.lower() in _err
                            )
                        )
                        if _using_partition:
                            sql_fallback = _default_sql(selected_table, row_limit, "", None)
                            st.warning(
                                f"⚠️ Coluna de partição **'{partition_col}'** não encontrada "
                                f"na tabela — query refeita sem particionamento."
                            )
                            df = athena.query(sql_fallback)
                        else:
                            raise

                st.session_state.df            = df
                st.session_state.data_is_sample = row_limit is not None

                if st.session_state.agent and not df.empty:
                    st.session_state.agent.system_prompt = _build_system_prompt(
                        df, selected_table, row_limit is not None
                    )

                if df.empty:
                    st.warning("Query retornou zero registros.")
                else:
                    st.success(f"✅ {len(df)} linhas × {len(df.columns)} colunas carregadas")
            except Exception as exc:
                st.error(f"Erro Athena: {exc}")

        if not st.session_state.df.empty:
            d = st.session_state.df
            st.info(f"📋 {len(d)} linhas × {len(d.columns)} colunas em memória")

    # ── Controles do chat ──────────────────────────────────────────────────────
    st.divider()
    if st.button("🗑️ Limpar conversa", use_container_width=True):
        st.session_state.messages = []
        st.session_state.context  = []
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.title(f"{_app['icon']} {_app['title']}")
st.caption(
    "Converse com seus dados de obsolescência usando **IA Generativa (IARA)**. "
    "Configure as credenciais no painel lateral, carregue os dados e faça perguntas."
)

# ── Status bar ────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Agente IA", "✅ Conectado" if st.session_state.agent else "⚠️ Desconectado")
with c2:
    data_ok = not st.session_state.df.empty
    st.metric("Dados carregados", f"{len(st.session_state.df)} registros" if data_ok else "⚠️ Nenhum")
with c3:
    n_turns = sum(1 for m in st.session_state.messages if m["role"] == "user")
    st.metric("Perguntas realizadas", str(n_turns))

st.divider()

# ── Data preview ──────────────────────────────────────────────────────────────
if not st.session_state.df.empty:
    df = st.session_state.df
    with st.expander(
        f"📋 [{st.session_state.selected_table}] — {len(df)} linhas × {len(df.columns)} colunas",
        expanded=False,
    ):
        st.dataframe(df, use_container_width=True, hide_index=True)
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols):
            st.caption("Estatísticas numéricas:")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
else:
    st.info(
        "👈 **Passo 1:** Configure o agente de IA e clique em **Conectar IA**.  \n"
        "👈 **Passo 2:** Clique em **Listar Tabelas**, selecione a tabela e clique em **Carregar Dados**.  \n"
        "💬 **Passo 3:** Faça suas perguntas no campo abaixo!"
    )

# ── Chat messages ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Faça uma pergunta sobre os dados..."):
    if not st.session_state.agent:
        st.warning("⚠️ Configure e conecte o agente de IA no painel lateral.")
        st.stop()
    if st.session_state.df.empty:
        st.warning("⚠️ Carregue os dados do Athena antes de fazer perguntas.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        try:
            stream = st.session_state.agent.ask(
                question=prompt,
                context=st.session_state.context or None,
                stream=True,
            )
            answer: str = st.write_stream(stream)
        except Exception as exc:
            answer = f"❌ Erro ao processar pergunta: {exc}"
            st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.context.extend([
        {"role": "user",      "content": prompt},
        {"role": "assistant", "content": answer},
    ])
