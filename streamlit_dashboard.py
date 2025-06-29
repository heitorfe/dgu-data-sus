import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Dashboard SUS - Análise de Preços",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache data loading
@st.cache_data
def load_data():
    """Load and cache the SUS data"""
    try:
        df = pd.read_csv('data/sus/sus_data_geo.csv')
        df['compra'] = pd.to_datetime(df['compra'])
        df['insercao'] = pd.to_datetime(df['insercao'])
        df['preco_unitario'] = pd.to_numeric(df['preco_unitario'], errors='coerce')
        df['preco_total'] =  pd.to_numeric(df['preco_total'], errors='coerce')
        return df
    except FileNotFoundError:
        st.error("Arquivo de dados não encontrado. Execute o notebook data_transform.ipynb primeiro.")
        return None

def format_currency(value):
    """Format value as Brazilian currency"""
    return f"R$ {value:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')

def calculate_savings_potential(df_filtered):
    """Calculate potential savings by using lowest prices in region"""
    savings_data = []
    
    for categoria in df_filtered['categoria'].unique():
        cat_data = df_filtered[df_filtered['categoria'] == categoria]
        
        if len(cat_data) < 2:
            continue
            
        # Group by region and calculate stats
        region_stats = cat_data.groupby('regiao').agg({
            'preco_unitario': ['min', 'mean', 'median', 'count'],
            'preco_total': 'sum',
            'qtd_itens_comprados': 'sum'
        }).round(2)
        
        region_stats.columns = ['preco_min', 'preco_medio', 'preco_mediano', 'count', 'gasto_total', 'qtd_total']
        
        for regiao in region_stats.index:
            preco_min = region_stats.loc[regiao, 'preco_min']
            preco_medio = region_stats.loc[regiao, 'preco_medio']
            gasto_total = region_stats.loc[regiao, 'gasto_total']
            qtd_total = region_stats.loc[regiao, 'qtd_total']
            
            if preco_min < preco_medio and qtd_total > 0:
                economia_potencial = (preco_medio - preco_min) * qtd_total
                economia_percentual = ((preco_medio - preco_min) / preco_medio) * 100
                
                savings_data.append({
                    'categoria': categoria,
                    'regiao': regiao,
                    'preco_min': preco_min,
                    'preco_medio': preco_medio,
                    'gasto_total': gasto_total,
                    'qtd_total': qtd_total,
                    'economia_potencial': economia_potencial,
                    'economia_percentual': economia_percentual
                })
    
    return pd.DataFrame(savings_data)

# Load data
df = load_data()
if df is None:
    st.stop()

# Sidebar filters
st.sidebar.header("Filtros Gerais")

# Date range filter
min_date = df['compra'].min()
max_date = df['compra'].max()
date_range = st.sidebar.date_input(
    "Período de Compra",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Apply date filter
if len(date_range) == 2:
    df_filtered = df[(df['compra'] >= pd.to_datetime(date_range[0])) & 
                     (df['compra'] <= pd.to_datetime(date_range[1]))]
else:
    df_filtered = df


# Additional filters with collapsible sections
with st.sidebar.expander("🔍 Filtros Avançados", expanded=False):
    st.markdown("### Filtros por Região e Categoria")
    
    # Enable/disable advanced filters
    use_advanced_filters = st.checkbox("Habilitar filtros avançados", value=False)
    
    if use_advanced_filters:
        selected_regions = st.multiselect(
            "Regiões",
            options=df_filtered['regiao'].unique(),
            default=df_filtered['regiao'].unique()
        )

        selected_macro_categories = st.multiselect(
            "Macro Categorias",
            options=df_filtered['macro_categoria'].unique(),
            default=df_filtered['macro_categoria'].unique()
        )

        # Filter dataframe based on selected macro categories to get relevant submacro categories
        df_macro_filtered = df_filtered[df_filtered['macro_categoria'].isin(selected_macro_categories)]

        selected_submacro_categories = st.multiselect(
            "Sub Categorias",
            options=df_macro_filtered['submacro_categoria'].unique(),
            default=df_macro_filtered['submacro_categoria'].unique()
        )

        # Filter again based on selected submacro categories to get relevant categories
        df_submacro_filtered = df_macro_filtered[df_macro_filtered['submacro_categoria'].isin(selected_submacro_categories)]

        selected_categories = st.multiselect(
            "Categorias Específicas",
            options=df_submacro_filtered['categoria'].unique(),
            default=df_submacro_filtered['categoria'].unique()
        )

        # Apply all filters
        df_filtered = df_filtered[
            (df_filtered['regiao'].isin(selected_regions)) &
            (df_filtered['macro_categoria'].isin(selected_macro_categories)) &
            (df_filtered['submacro_categoria'].isin(selected_submacro_categories)) &
            (df_filtered['categoria'].isin(selected_categories))
        ]
    else:
        st.info("Filtros avançados desabilitados - mostrando todos os dados")



# Main dashboard
st.title("🏥 Dashboard SUS - Análise de Preços e Compras")
st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Visão Geral", 
    "📈 Variação de Preços", 
    "🗺️ Análise Regional", 
    "💰 Economia Potencial"
])

with tab1:
    st.header("Visão Geral das Compras")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_gasto = df_filtered['preco_total'].sum()
        if total_gasto >= 1_000_000:
            st.metric("Gasto Total", f"R$ {total_gasto/1_000_000:.1f}M")
        else:
            st.metric("Gasto Total", format_currency(total_gasto))
    
    with col2:
        total_itens = df_filtered['qtd_itens_comprados'].sum()
        if total_itens >= 1_000_000:
            st.metric("Itens Comprados", f"{total_itens/1_000_000:.1f}M")
        else:
            st.metric("Itens Comprados", f"{total_itens:,.0f}".replace(',', '.'))
    
    with col3:
        avg_price = df_filtered['preco_unitario'].mean()
        st.metric("Preço Médio", format_currency(avg_price))
    
    with col4:
        num_suppliers = df_filtered['fornecedor'].nunique()
        st.metric("Fornecedores", f"{num_suppliers:,}".replace(',', '.'))
    
    # Top categories by spending
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Categorias por Gasto")
        top_categories = df_filtered.groupby('categoria')['preco_total'].sum().nlargest(10)
        fig = px.bar(
            x=top_categories.values,
            y=top_categories.index,
            orientation='h',
            title="Gasto por Categoria"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Distribuição por Região")
        region_spending = df_filtered.groupby('regiao')['preco_total'].sum()
        fig = px.pie(
            values=region_spending.values,
            names=region_spending.index,
            title="Gasto por Região"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Variação de Preços ao Longo do Tempo")
    
    # Select category for analysis
    selected_cat = st.selectbox(
        "Selecione uma categoria para análise temporal:",
        options=df_filtered['categoria'].unique(),
        index=0
    )
    
    cat_data = df_filtered[df_filtered['categoria'] == selected_cat]
    
    if not cat_data.empty:
        # Monthly price evolution
        monthly_prices = cat_data.groupby(cat_data['compra'].dt.to_period('M')).agg({
            'preco_unitario': ['mean', 'median', 'min', 'max'],
            'preco_total': 'sum',
            'qtd_itens_comprados': 'sum'
        }).round(2)
        
        monthly_prices.index = monthly_prices.index.to_timestamp()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price evolution chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly_prices.index,
                y=monthly_prices[('preco_unitario', 'mean')],
                mode='lines+markers',
                name='Preço Médio',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=monthly_prices.index,
                y=monthly_prices[('preco_unitario', 'median')],
                mode='lines+markers',
                name='Preço Mediano',
                line=dict(color='red')
            ))
            fig.update_layout(
                title=f"Evolução de Preços - {selected_cat}",
                xaxis_title="Período",
                yaxis_title="Preço (R$)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Volume evolution
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=monthly_prices.index,
                y=monthly_prices[('qtd_itens_comprados', 'sum')],
                name='Quantidade Comprada'
            ))
            fig.update_layout(
                title=f"Volume de Compras - {selected_cat}",
                xaxis_title="Período",
                yaxis_title="Quantidade",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Price volatility by supplier
        st.subheader("Variação de Preços por Fornecedor")
        supplier_stats = cat_data.groupby('fornecedor').agg({
            'preco_unitario': ['mean', 'std', 'count'],
            'preco_total': 'sum'
        }).round(2)
        
        supplier_stats.columns = ['preco_medio', 'desvio_padrao', 'num_compras', 'gasto_total']
        supplier_stats = supplier_stats[supplier_stats['num_compras'] >= 5].nlargest(15, 'gasto_total')
        
        if not supplier_stats.empty:
            fig = px.scatter(
                supplier_stats.reset_index(),
                x='preco_medio',
                y='desvio_padrao',
                size='gasto_total',
                hover_data=['num_compras'],
                title="Preço Médio vs Volatilidade por Fornecedor"
            )
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Análise Regional")
    
    # UF selector
    selected_uf = st.selectbox(
        "Selecione um Estado (UF):",
        options=sorted(df_filtered['uf'].unique()),
        index=0
    )
    
    uf_data = df_filtered[df_filtered['uf'] == selected_uf]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Top 15 Municípios por Gasto - {selected_uf}")
        
        municipal_spending = uf_data.groupby('municipio_instituicao').agg({
            'preco_total': 'sum',
            'qtd_itens_comprados': 'sum',
            'nome_instituicao': 'nunique'
        }).round(2)
        
        municipal_spending.columns = ['gasto_total', 'qtd_total', 'num_instituicoes']
        top_municipalities = municipal_spending.nlargest(15, 'gasto_total')
        
        fig = px.bar(
            top_municipalities.reset_index(),
            x='gasto_total',
            y='municipio_instituicao',
            orientation='h',
            title=f"Gasto por Município - {selected_uf}",
            hover_data=['qtd_total', 'num_instituicoes']
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Comparação de Preços por Região")
        
        # Box plot of prices by region for selected category
        comp_category = st.selectbox(
            "Categoria para comparação:",
            options=df_filtered['categoria'].unique(),
            key="regional_category"
        )
        
        regional_data = df_filtered[df_filtered['categoria'] == comp_category]
        
        if not regional_data.empty:
            fig = px.box(
                regional_data,
                x='regiao',
                y='preco_unitario',
                title=f"Distribuição de Preços por Região - {comp_category}"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Regional price comparison table
            regional_stats = regional_data.groupby('regiao')['preco_unitario'].agg([
                'mean', 'median', 'std', 'min', 'max', 'count'
            ]).round(2)
            regional_stats.columns = ['Média', 'Mediana', 'Desvio Padrão', 'Mínimo', 'Máximo', 'Compras']
            
            st.subheader("Estatísticas por Região")
            st.dataframe(regional_stats)

with tab4:
    st.header("Análise de Economia Potencial")
    
    st.markdown("""
    Esta análise identifica oportunidades de economia comparando preços dentro de cada região.
    A economia potencial é calculada considerando o que seria economizado se todas as compras 
    fossem feitas pelo menor preço da região.
    """)
    
    # Calculate savings
    savings_df = calculate_savings_potential(df_filtered)
    
    if not savings_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top categories with highest savings potential
            top_savings = savings_df.nlargest(15, 'economia_potencial')
            
            fig = px.bar(
                top_savings,
                x='economia_potencial',
                y='categoria',
                color='regiao',
                orientation='h',
                title="Top 15 - Maior Potencial de Economia"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Savings percentage analysis
            fig = px.scatter(
                savings_df,
                x='gasto_total',
                y='economia_percentual',
                size='economia_potencial',
                color='regiao',
                hover_data=['categoria'],
                title="Gasto Total vs % de Economia Potencial"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed savings table
        st.subheader("Detalhamento da Economia Potencial")
        
        # Region filter for savings
        selected_region_savings = st.selectbox(
            "Filtrar por região:",
            options=['Todas'] + list(savings_df['regiao'].unique()),
            key="savings_region"
        )
        
        if selected_region_savings != 'Todas':
            display_savings = savings_df[savings_df['regiao'] == selected_region_savings]
        else:
            display_savings = savings_df
        
        display_savings = display_savings.sort_values('economia_potencial', ascending=False)
        
        # Format the display table
        display_table = display_savings.copy()
        display_table['preco_min'] = display_table['preco_min'].apply(format_currency)
        display_table['preco_medio'] = display_table['preco_medio'].apply(format_currency)
        display_table['gasto_total'] = display_table['gasto_total'].apply(format_currency)
        display_table['economia_potencial'] = display_table['economia_potencial'].apply(format_currency)
        display_table['economia_percentual'] = display_table['economia_percentual'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(
            display_table[['categoria', 'regiao', 'preco_min', 'preco_medio', 
                          'gasto_total', 'economia_potencial', 'economia_percentual']].rename(columns={
                'categoria': 'Categoria',
                'regiao': 'Região',
                'preco_min': 'Preço Mínimo',
                'preco_medio': 'Preço Médio',
                'gasto_total': 'Gasto Total',
                'economia_potencial': 'Economia Potencial',
                'economia_percentual': '% Economia'
            }),
            use_container_width=True
        )
        
        # Summary metrics
        total_savings = savings_df['economia_potencial'].sum()
        total_spending = savings_df['gasto_total'].sum()
        avg_savings_pct = (total_savings / total_spending) * 100 if total_spending > 0 else 0
        
        st.markdown("### Resumo da Economia Potencial")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Economia Total Potencial", format_currency(total_savings))
        with col2:
            st.metric("Gasto Total Analisado", format_currency(total_spending))
        with col3:
            st.metric("% Economia Média", f"{avg_savings_pct:.1f}%")
    
    else:
        st.warning("Não foi possível calcular economia potencial com os filtros atuais.")

# Footer
st.markdown("---")
st.markdown("Dashboard desenvolvido para análise de dados do Sistema Único de Saúde (SUS)")
