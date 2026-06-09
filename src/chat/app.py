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

def _default_sql(table: str) -> str:
    """Gera o SQL padrão. Sem LIMIT quando default_limit está vazio."""
    limit_val = str(_ath.get("default_limit", "")).strip()
    limit_clause = f" LIMIT {limit_val}" if limit_val.isdigit() else ""
    return f'SELECT * FROM "{_ath["database"]}"."{table}"{limit_clause};'


def _build_system_prompt(df: pd.DataFrame, table: str) -> str:
    cols     = "\n".join(f"  - `{c}`" for c in df.columns)
    table_md = df.to_markdown(index=False) if not df.empty else "_(vazio)_"
    return f"""Você é um assistente analítico especializado em **obsolescência de servidores** do Itaú Unibanco.

Você tem acesso a dados da tabela `{table}` no AWS Athena \
(banco: `{_ath["database"]}`).

## Colunas disponíveis
{cols}

## Dados carregados
{table_md}

**Diretrizes:**
- Responda sempre em **português brasileiro**.
- Baseie suas análises nos dados acima; não invente valores.
- Use tabelas markdown e números formatados quando útil.
- Se a pergunta exigir dados além da amostra carregada, informe claramente e sugira uma query SQL para obtê-los.
- Seja objetivo, analítico e preciso.
"""


def _make_agent(
    provider: str, api_key: str, client_id: str, client_secret: str,
    model: str, temperature: float, system_prompt: str = "",
) -> IaraAgentChat:
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
    "messages":         [],
    "context":          [],
    "df":               pd.DataFrame(),
    "agent":            None,
    "available_tables": [],          # lista de tabelas do Glue catalog
    "selected_table":   _ath["table"],
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
                        _build_system_prompt(st.session_state.df, st.session_state.selected_table)
                        if not st.session_state.df.empty else ""
                    )
                    st.session_state.agent = _make_agent(
                        provider, api_key, client_id, client_secret, model, temperature, sys_prompt)
                st.success("Agente conectado!")
            except Exception as exc:
                st.error(f"Erro ao conectar: {exc}")

        st.success("✅ IA conectada") if st.session_state.agent else st.info("ℹ️ Clique em **Conectar IA**.")

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

        # ── Seleção de tabela ──────────────────────────────────────────────────
        if st.button("🔍 Listar Tabelas", use_container_width=True):
            try:
                with st.spinner("Buscando tabelas no Glue Catalog..."):
                    tmp = _make_athena(aws_region, s3_output.strip(), _ath["table"],
                                       workgroup, aws_key, aws_secret, aws_token, aws_profile)
                    st.session_state.available_tables = tmp.list_tables()
                st.success(f"{len(st.session_state.available_tables)} tabelas encontradas.")
            except Exception as exc:
                st.error(f"Erro ao listar tabelas: {exc}")

        # Combobox — usa tabelas do Glue se já listadas, senão só o default do config
        table_options = st.session_state.available_tables or [_ath["table"]]
        current_table = st.session_state.selected_table
        default_idx   = table_options.index(current_table) if current_table in table_options else 0

        selected_table = st.selectbox(
            "Tabela",
            table_options,
            index=default_idx,
            help="Clique em **Listar Tabelas** para carregar todas as tabelas do database.",
        )
        st.session_state.selected_table = selected_table

        # SQL customizado — placeholder reflete a tabela selecionada
        custom_sql = st.text_area(
            "SQL customizado (opcional)",
            placeholder=_default_sql(selected_table),
            height=90,
            help="Deixe em branco para usar a query padrão (com LIMIT do settings.toml, "
                 "ou sem LIMIT se default_limit estiver vazio).",
        )

        if st.button("📊 Carregar Dados", use_container_width=True, type="primary"):
            try:
                with st.spinner("Consultando Athena..."):
                    athena = _make_athena(aws_region, s3_output.strip(), selected_table,
                                          workgroup, aws_key, aws_secret, aws_token, aws_profile)
                    sql = custom_sql.strip() or _default_sql(selected_table)
                    df  = athena.query(sql)

                st.session_state.df = df

                if st.session_state.agent and not df.empty:
                    st.session_state.agent.system_prompt = _build_system_prompt(df, selected_table)

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
