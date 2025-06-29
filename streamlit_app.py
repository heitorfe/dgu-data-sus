import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from datetime import datetime, timedelta
import io

# Configuração da página
st.set_page_config(
    page_title="Dashboard SUS - Análise de Preços",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache para carregar dados
@st.cache_data
def load_data():
    """Carrega e prepara os dados"""
    try:
        df = pd.read_csv('data/sus/sus_data_geo.csv')
        
        # Mapeamento das regiões brasileiras
        region_mapping = {
            "Norte": ["AC", "AM", "AP", "PA", "RO", "RR", "TO"],
            "Nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
            "Centro-Oeste": ["DF", "GO", "MT", "MS"],
            "Sudeste": ["ES", "MG", "RJ", "SP"],
            "Sul": ["PR", "RS", "SC"]
        }
        
        df['regiao'] = df['uf'].apply(
            lambda x: next((region for region, states in region_mapping.items() if x in states), None)
        )
        
        # Conversões e limpeza
        df['compra'] = pd.to_datetime(df['compra'], errors='coerce')
        df['valor_total'] = df['preco_unitario'] * df['qtd_itens_comprados']
        df['ano'] = df['compra'].dt.year
        df['mes'] = df['compra'].dt.month
        df['trimestre'] = df['compra'].dt.quarter
        
        # Limpeza básica
        df = df[df['preco_unitario'] < 30_000]
        df = df[df['qtd_itens_comprados'] < 10_000_000]
        df = df.dropna(subset=['regiao', 'compra'])
        
        # Criar categorias de produtos
        df['categoria_produto'] = df['descricao_catmat'].apply(classify_product_category)
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()

def classify_product_category(description):
    """Classifica produtos em categorias"""
    if pd.isna(description):
        return "Não classificado"
    
    desc_lower = str(description).lower()
    
    # Categorias de produtos médicos
    categories = {
        'Medicamentos': ['medicamento', 'remedio', 'comprimido', 'capsula', 'ampola', 'frasco'],
        'Equipamentos': ['equipamento', 'aparelho', 'monitor', 'ventilador', 'bomba'],
        'Materiais Cirúrgicos': ['cirurgic', 'bisturi', 'agulha', 'seringa', 'cateter'],
        'Insumos': ['insumo', 'material', 'descartavel', 'luva', 'mascara'],
        'Implantes': ['implante', 'protese', 'stent', 'valvula'],
        'Diagnóstico': ['teste', 'kit', 'reagente', 'exame']
    }
    
    for category, keywords in categories.items():
        if any(keyword in desc_lower for keyword in keywords):
            return category
    
    return "Outros"

@st.cache_data
def calculate_price_discrepancies(df_filtered):
    """Calcula discrepâncias de preços"""
    discrepancies = []
    
    for codigo in df_filtered['codigo_br'].unique():
        product_data = df_filtered[df_filtered['codigo_br'] == codigo]
        
        if len(product_data) < 5:  # Mínimo de compras para análise
            continue
            
        regional_stats = product_data.groupby('regiao')['preco_unitario'].agg([
            'count', 'median', 'std'
        ]).round(2)
        
        if len(regional_stats) >= 2:
            min_median = regional_stats['median'].min()
            max_median = regional_stats['median'].max()
            price_ratio = max_median / min_median if min_median > 0 else 0
            
            discrepancies.append({
                'codigo_br': codigo,
                'descricao': product_data['descricao_catmat'].iloc[0],
                'categoria': product_data['categoria_produto'].iloc[0],
                'price_ratio': price_ratio,
                'min_price': min_median,
                'max_price': max_median,
                'total_purchases': len(product_data),
                'regions_count': len(regional_stats)
            })
    
    return pd.DataFrame(discrepancies).sort_values('price_ratio', ascending=False)

def main():
    st.title("💊 Dashboard SUS - Análise Comparativa de Preços")
    st.markdown("---")
    
    # Carregar dados
    df = load_data()
    
    if df.empty:
        st.error("Não foi possível carregar os dados. Verifique se o arquivo existe.")
        return
    
    # Sidebar com filtros
    st.sidebar.header("🔍 Filtros")
    
    # Filtro de período
    st.sidebar.subheader("Período")
    min_date = df['compra'].min().date()
    max_date = df['compra'].max().date()
    
    date_range = st.sidebar.date_input(
        "Selecione o período:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filtro de região
    st.sidebar.subheader("Regiões")
    regioes_disponiveis = ['Todas'] + sorted(df['regiao'].unique().tolist())
    regioes_selecionadas = st.sidebar.multiselect(
        "Selecione as regiões:",
        regioes_disponiveis,
        default=['Todas']
    )
    
    # Filtro de categoria
    st.sidebar.subheader("Categorias")
    categorias_disponiveis = ['Todas'] + sorted(df['categoria_produto'].unique().tolist())
    categorias_selecionadas = st.sidebar.multiselect(
        "Selecione as categorias:",
        categorias_disponiveis,
        default=['Todas']
    )
    
    # Filtro de fornecedor
    st.sidebar.subheader("Fornecedores")
    top_fornecedores = df['fornecedor'].value_counts().head(20).index.tolist()
    fornecedores_disponiveis = ['Todos'] + top_fornecedores
    fornecedores_selecionados = st.sidebar.multiselect(
        "Top 20 fornecedores:",
        fornecedores_disponiveis,
        default=['Todos']
    )
    
    # Aplicar filtros
    df_filtered = df.copy()
    
    # Filtro de data
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df_filtered[
            (df_filtered['compra'].dt.date >= start_date) & 
            (df_filtered['compra'].dt.date <= end_date)
        ]
    
    # Filtro de região
    if 'Todas' not in regioes_selecionadas and regioes_selecionadas:
        df_filtered = df_filtered[df_filtered['regiao'].isin(regioes_selecionadas)]
    
    # Filtro de categoria
    if 'Todas' not in categorias_selecionadas and categorias_selecionadas:
        df_filtered = df_filtered[df_filtered['categoria_produto'].isin(categorias_selecionadas)]
    
    # Filtro de fornecedor
    if 'Todos' not in fornecedores_selecionados and fornecedores_selecionados:
        df_filtered = df_filtered[df_filtered['fornecedor'].isin(fornecedores_selecionados)]
    
    # Verificar se há dados após filtros
    if df_filtered.empty:
        st.warning("Nenhum dado encontrado com os filtros selecionados.")
        return
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total de Registros",
            f"{len(df_filtered):,}",
            delta=f"{len(df_filtered) - len(df):,}"
        )
    
    with col2:
        st.metric(
            "Produtos Únicos",
            f"{df_filtered['codigo_br'].nunique():,}"
        )
    
    with col3:
        st.metric(
            "Valor Total",
            f"R$ {df_filtered['valor_total'].sum()/1e6:.1f}M"
        )
    
    with col4:
        st.metric(
            "Preço Médio",
            f"R$ {df_filtered['preco_unitario'].mean():.2f}"
        )
    
    st.markdown("---")
    
    # Tabs para diferentes análises
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Visão Geral", 
        "🗺️ Análise Regional", 
        "⏰ Análise Temporal", 
        "🏭 Análise de Fornecedores",
        "🔍 Discrepâncias"
    ])
    
    with tab1:
        st.header("📊 Visão Geral")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribuição por categoria
            cat_data = df_filtered['categoria_produto'].value_counts()
            fig_cat = px.pie(
                values=cat_data.values, 
                names=cat_data.index,
                title="Distribuição por Categoria de Produto"
            )
            st.plotly_chart(fig_cat, use_container_width=True)
        
        with col2:
            # Distribuição por região
            reg_data = df_filtered['regiao'].value_counts()
            fig_reg = px.bar(
                x=reg_data.index, 
                y=reg_data.values,
                title="Número de Compras por Região"
            )
            fig_reg.update_layout(xaxis_title="Região", yaxis_title="Número de Compras")
            st.plotly_chart(fig_reg, use_container_width=True)
        
        # Produtos com maiores variações de preço
        st.subheader("🎯 Produtos com Maiores Variações de Preço")
        
        price_variations = []
        for codigo in df_filtered['codigo_br'].value_counts().head(50).index:
            product_data = df_filtered[df_filtered['codigo_br'] == codigo]
            if len(product_data) >= 5:
                min_price = product_data['preco_unitario'].min()
                max_price = product_data['preco_unitario'].max()
                ratio = max_price / min_price if min_price > 0 else 0
                
                price_variations.append({
                    'Produto': product_data['descricao_catmat'].iloc[0],
                    'Categoria': product_data['categoria_produto'].iloc[0],
                    'Preço Mín': f"R$ {min_price:.2f}",
                    'Preço Máx': f"R$ {max_price:.2f}",
                    'Razão': f"{ratio:.1f}x",
                    'Compras': len(product_data)
                })
        
        if price_variations:
            variations_df = pd.DataFrame(price_variations)
            variations_df = variations_df.sort_values('Razão', ascending=False, key=lambda x: x.str.replace('x', '').astype(float))
            st.dataframe(variations_df.head(15), use_container_width=True)
    
    with tab2:
        st.header("🗺️ Análise Regional")
        
        # Comparação de preços por região
        regional_comparison = df_filtered.groupby(['regiao', 'categoria_produto'])['preco_unitario'].agg([
            'count', 'mean', 'median'
        ]).round(2).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Preço médio por região
            regional_avg = df_filtered.groupby('regiao')['preco_unitario'].mean().reset_index()
            fig_reg_avg = px.bar(
                regional_avg, 
                x='regiao', 
                y='preco_unitario',
                title="Preço Médio por Região",
                color='preco_unitario',
                color_continuous_scale='viridis'
            )
            fig_reg_avg.update_layout(xaxis_title="Região", yaxis_title="Preço Médio (R$)")
            st.plotly_chart(fig_reg_avg, use_container_width=True)
        
        with col2:
            # Volume financeiro por região
            regional_volume = df_filtered.groupby('regiao')['valor_total'].sum().reset_index()
            fig_reg_vol = px.bar(
                regional_volume, 
                x='regiao', 
                y='valor_total',
                title="Volume Financeiro por Região",
                color='valor_total',
                color_continuous_scale='blues'
            )
            fig_reg_vol.update_layout(xaxis_title="Região", yaxis_title="Volume (R$)")
            st.plotly_chart(fig_reg_vol, use_container_width=True)
        
        # Heatmap de preços por região e categoria
        if len(categorias_selecionadas) > 1 or 'Todas' in categorias_selecionadas:
            pivot_data = df_filtered.groupby(['regiao', 'categoria_produto'])['preco_unitario'].median().unstack(fill_value=0)
            
            fig_heatmap = px.imshow(
                pivot_data.values,
                labels=dict(x="Categoria", y="Região", color="Preço Mediano"),
                x=pivot_data.columns,
                y=pivot_data.index,
                title="Heatmap: Preço Mediano por Região e Categoria",
                color_continuous_scale='RdYlBu_r'
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Análise específica por produto
        st.subheader("🔍 Análise Específica por Produto")
        
        # Adicionar cache para operações custosas
        @st.cache_data
        def get_top_produtos(df_filtered):
            """Cache da lista de produtos mais frequentes"""
            return df_filtered['descricao_catmat'].value_counts().head(20).index.tolist()

        @st.cache_data
        def filter_produto_data(df_filtered, produto_selecionado):
            """Cache dos dados filtrados por produto"""
            return df_filtered[df_filtered['descricao_catmat'] == produto_selecionado]

        produtos_disponiveis = get_top_produtos(df_filtered)
        produto_selecionado = st.selectbox(
            "Selecione um produto para análise detalhada:",
            produtos_disponiveis
        )
        
        if produto_selecionado:
            produto_data = filter_produto_data(df_filtered, produto_selecionado)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Boxplot por região
                fig_box = px.box(
                    produto_data, 
                    x='regiao', 
                    y='preco_unitario',
                    title=f"Distribuição de Preços por Região\n{produto_selecionado}..."
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            with col2:
                # Estatísticas por região
                produto_stats = produto_data.groupby('regiao')['preco_unitario'].agg([
                    'count', 'mean', 'median', 'std'
                ]).round(2).reset_index()
                produto_stats.columns = ['Região', 'Compras', 'Média', 'Mediana', 'Desvio Padrão']
                st.dataframe(produto_stats, use_container_width=True)
    
    with tab3:
        st.header("⏰ Análise Temporal")
        
        # # Evolução temporal dos preços
        # temporal_data = df_filtered.groupby([
        #     df_filtered['compra'].dt.to_period('M'), 'regiao'
        # ])['preco_unitario'].median().reset_index()
        # temporal_data['compra'] = temporal_data['compra'].astype(str)
        
        # fig_temporal = px.line(
        #     temporal_data, 
        #     x='compra', 
        #     y='preco_unitario', 
        #     color='regiao',
        #     title="Evolução Temporal dos Preços Medianos por Região",
        #     markers=True
        # )
        # fig_temporal.update_layout(xaxis_title="Período", yaxis_title="Preço Mediano (R$)")
        # st.plotly_chart(fig_temporal, use_container_width=True)
        
        # col1, col2 = st.columns(2)
        
        # with col1:
        #     # Sazonalidade por mês
        #     monthly_data = df_filtered.groupby(df_filtered['compra'].dt.month)['preco_unitario'].mean().reset_index()
        #     monthly_data['mes_nome'] = monthly_data['compra'].map({
        #         1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
        #         7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'
        #     })
            
        #     fig_seasonal = px.bar(
        #         monthly_data, 
        #         x='mes_nome', 
        #         y='preco_unitario',
        #         title="Sazonalidade - Preço Médio por Mês"
        #     )
        #     st.plotly_chart(fig_seasonal, use_container_width=True)
        
        # with col2:
        #     # Volume por trimestre - Create temporary DataFrame to avoid column conflicts
        #     quarterly_temp = df_filtered[['compra', 'valor_total']].copy()
        #     quarterly_temp['ano_ref'] = quarterly_temp['compra'].dt.year
        #     quarterly_temp['trimestre_ref'] = quarterly_temp['compra'].dt.quarter
            
        #     quarterly_data = quarterly_temp.groupby(['ano_ref', 'trimestre_ref'])['valor_total'].sum().reset_index()
        #     quarterly_data['periodo'] = quarterly_data['ano_ref'].astype(str) + '-Q' + quarterly_data['trimestre_ref'].astype(str)
            
        #     fig_quarterly = px.bar(
        #         quarterly_data, 
        #         x='periodo', 
        #         y='valor_total',
        #         title="Volume Financeiro por Trimestre"
        #     )
        #     fig_quarterly.update_layout(xaxis_title="Período", yaxis_title="Volume (R$)")
        #     st.plotly_chart(fig_quarterly, use_container_width=True)
    
    with tab4:
        st.header("🏭 Análise de Fornecedores")
        
        # Top fornecedores
        top_suppliers = df_filtered.groupby('fornecedor').agg({
            'valor_total': 'sum',
            'preco_unitario': ['count', 'mean'],
            'codigo_br': 'nunique'
        }).round(2)
        
        top_suppliers.columns = ['Valor Total', 'Num Compras', 'Preço Médio', 'Produtos Únicos']
        top_suppliers = top_suppliers.sort_values('Valor Total', ascending=False).head(15)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top 10 fornecedores por volume
            top_10_suppliers = top_suppliers.head(10).reset_index()
            fig_suppliers = px.bar(
                top_10_suppliers, 
                x='Valor Total', 
                y='fornecedor',
                title="Top 10 Fornecedores por Volume Financeiro",
                orientation='h'
            )
            st.plotly_chart(fig_suppliers, use_container_width=True)
        
        with col2:
            # Dispersão: Preço médio vs Volume
            scatter_data = top_suppliers.reset_index()
            fig_scatter = px.scatter(
                scatter_data, 
                x='Preço Médio', 
                y='Valor Total',
                size='Num Compras',
                hover_data=['fornecedor'],
                title="Preço Médio vs Volume Financeiro",
                log_x=True,
                log_y=True
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Análise de variação de preços por fornecedor
        st.subheader("📊 Variação de Preços por Fornecedor")
        
        supplier_variations = []
        for fornecedor in df_filtered['fornecedor'].value_counts().head(20).index:
            supplier_data = df_filtered[df_filtered['fornecedor'] == fornecedor]
            if len(supplier_data) >= 10:
                cv = supplier_data['preco_unitario'].std() / supplier_data['preco_unitario'].mean()
                supplier_variations.append({
                    'Fornecedor': fornecedor + '...',
                    'Compras': len(supplier_data),
                    'Preço Médio': f"R$ {supplier_data['preco_unitario'].mean():.2f}",
                    'Coef. Variação': f"{cv:.2f}",
                    'Produtos': supplier_data['codigo_br'].nunique()
                })
        
        if supplier_variations:
            variations_df = pd.DataFrame(supplier_variations)
            variations_df = variations_df.sort_values('Coef. Variação', ascending=False, key=lambda x: x.astype(float))
            st.dataframe(variations_df, use_container_width=True)
    
    with tab5:
        st.header("🔍 Análise de Discrepâncias")
        
        # Calcular discrepâncias
        discrepancies = calculate_price_discrepancies(df_filtered)
        
        if not discrepancies.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribuição das razões de preço
                fig_ratio_dist = px.histogram(
                    discrepancies, 
                    x='price_ratio',
                    nbins=30,
                    title="Distribuição das Razões de Preço (Max/Min)",
                    labels={'price_ratio': 'Razão de Preço', 'count': 'Frequência'}
                )
                st.plotly_chart(fig_ratio_dist, use_container_width=True)
            
            with col2:
                # Top discrepâncias por categoria
                top_discrepancies = discrepancies.head(20).copy()
                top_discrepancies['descricao_truncada'] = top_discrepancies['descricao']
                
                fig_top_disc = px.bar(
                    top_discrepancies, 
                    x='price_ratio', 
                    y='descricao_truncada',
                    color='categoria',
                    title="Top 20 Produtos com Maiores Discrepâncias",
                    orientation='h',
                    hover_data={'descricao': True, 'descricao_truncada': False}
                )
                fig_top_disc.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_top_disc, use_container_width=True)
            
            # Tabela detalhada das discrepâncias
            st.subheader("📋 Detalhamento das Discrepâncias")
            
            # Filtros para a tabela
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_ratio = st.number_input("Razão mínima:", min_value=1.0, value=2.0, step=0.1)
            
            with col2:
                min_purchases = st.number_input("Mínimo de compras:", min_value=1, value=10, step=1)
            
            with col3:
                categoria_filter = st.selectbox(
                    "Filtrar por categoria:",
                    ['Todas'] + sorted(discrepancies['categoria'].unique().tolist())
                )
            
            # Aplicar filtros na tabela
            filtered_discrepancies = discrepancies[
                (discrepancies['price_ratio'] >= min_ratio) &
                (discrepancies['total_purchases'] >= min_purchases)
            ]
            
            if categoria_filter != 'Todas':
                filtered_discrepancies = filtered_discrepancies[
                    filtered_discrepancies['categoria'] == categoria_filter
                ]
            
            # Formatar dados para exibição
            display_discrepancies = filtered_discrepancies.copy()
            display_discrepancies['Produto'] = display_discrepancies['descricao']
            display_discrepancies['Categoria'] = display_discrepancies['categoria']
            display_discrepancies['Razão de Preço'] = display_discrepancies['price_ratio'].round(1).astype(str) + 'x'
            display_discrepancies['Preço Mín'] = 'R$ ' + display_discrepancies['min_price'].round(2).astype(str)
            display_discrepancies['Preço Máx'] = 'R$ ' + display_discrepancies['max_price'].round(2).astype(str)
            display_discrepancies['Total Compras'] = display_discrepancies['total_purchases']
            display_discrepancies['Regiões'] = display_discrepancies['regions_count']
            
            st.dataframe(
                display_discrepancies[['Produto', 'Categoria', 'Razão de Preço', 'Preço Mín', 'Preço Máx', 'Total Compras', 'Regiões']],
                use_container_width=True
            )
            
            # Oportunidades de economia
            st.subheader("💰 Oportunidades de Economia")
            
            # Explicação do cálculo
            with st.expander("ℹ️ Como a economia é calculada?", expanded=False):
                st.markdown("""
                **Metodologia de Cálculo:**
                
                1. **Identificação do Benchmark**: Para cada produto, identificamos o menor preço mediano entre todas as regiões
                2. **Cálculo da Diferença**: Comparamos o preço de cada região com o benchmark (menor preço)
                3. **Projeção da Economia**: Multiplicamos a diferença de preço pela quantidade total comprada na região
                
                **Fórmula:**
                ```
                Economia = (Preço Atual - Preço Benchmark) × Quantidade Comprada
                ```
                
                **Exemplo Prático:**
                - Produto X: Região A paga R$ 10,00 / Região B paga R$ 6,00 (benchmark)
                - Região A comprou 1.000 unidades
                - Economia potencial para Região A: (R$ 10,00 - R$ 6,00) × 1.000 = R$ 4.000,00
                
                ⚠️ **Importante**: Esta é uma estimativa teórica que assume que seria possível aplicar o menor preço encontrado em todas as regiões.
                """)
            
            total_savings = 0
            savings_opportunities = []
            
            for _, row in filtered_discrepancies.head(20).iterrows():
                codigo = row['codigo_br']
                product_data = df_filtered[df_filtered['codigo_br'] == codigo]
                
                regional_stats = product_data.groupby('regiao').agg({
                    'preco_unitario': 'median',
                    'qtd_itens_comprados': 'sum'
                })
                
                if len(regional_stats) >= 2:
                    benchmark_price = regional_stats['preco_unitario'].min()
                    benchmark_region = regional_stats['preco_unitario'].idxmin()
                    
                    for regiao, stats in regional_stats.iterrows():
                        if stats['preco_unitario'] > benchmark_price:
                            savings_per_unit = stats['preco_unitario'] - benchmark_price
                            total_quantity = stats['qtd_itens_comprados']
                            potential_savings = savings_per_unit * total_quantity
                            total_savings += potential_savings
                            
                            savings_opportunities.append({
                                'Produto': row['descricao'],
                                'Região Atual': regiao,
                                'Preço Atual': f"R$ {stats['preco_unitario']:.2f}",
                                'Preço Benchmark': f"R$ {benchmark_price:.2f}",
                                'Região Benchmark': benchmark_region,
                                'Economia/Unidade': f"R$ {savings_per_unit:.2f}",
                                'Quantidade': f"{total_quantity:,}",
                                'Economia Total': f"R$ {potential_savings:,.2f}"
                            })
            
            if savings_opportunities:
                savings_df = pd.DataFrame(savings_opportunities)
                savings_df = savings_df.sort_values('Economia Total', ascending=False, key=lambda x: x.str.replace('R$ ', '').str.replace(',', '').astype(float))
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown("**Top 15 Oportunidades de Economia por Região:**")
                    st.dataframe(savings_df.head(15), use_container_width=True)
                
                with col2:
                    st.metric(
                        "💰 Economia Potencial Total",
                        f"R$ {total_savings:,.2f}",
                        help="Soma de todas as economias potenciais dos top 20 produtos com maiores discrepâncias"
                    )
                    
                    # Estatísticas adicionais
                    st.markdown("---")
                    st.markdown("**📊 Estatísticas:**")
                    st.markdown(f"🎯 **Produtos analisados**: {len(filtered_discrepancies.head(20))}")
                    st.markdown(f"🏢 **Oportunidades identificadas**: {len(savings_opportunities)}")
                    avg_savings = total_savings / len(savings_opportunities) if savings_opportunities else 0
                    st.markdown(f"💡 **Economia média por oportunidade**: R$ {avg_savings:,.2f}")
                
                # Resumo por região
                st.markdown("---")
                st.subheader("📍 Resumo de Economia por Região")
                
                region_summary = savings_df.groupby('Região Atual').agg({
                    'Economia Total': lambda x: sum(float(val.replace('R$ ', '').replace(',', '')) for val in x),
                    'Produto': 'count'  # Conta quantas oportunidades por região
                }).reset_index()
                region_summary.columns = ['Região', 'Economia Total', 'Oportunidades']
                region_summary['Economia Total'] = region_summary['Economia Total'].apply(lambda x: f"R$ {x:,.2f}")
                region_summary = region_summary.sort_values('Economia Total', ascending=False, key=lambda x: x.str.replace('R$ ', '').str.replace(',', '').astype(float))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(region_summary, use_container_width=True, hide_index=True)
                
                with col2:
                    # Gráfico de barras da economia por região
                    region_chart_data = region_summary.copy()
                    region_chart_data['Economia_Numerica'] = region_chart_data['Economia Total'].str.replace('R$ ', '').str.replace(',', '').astype(float)
                    
                    fig_region_savings = px.bar(
                        region_chart_data,
                        x='Região',
                        y='Economia_Numerica',
                        title="Economia Potencial por Região",
                        color='Economia_Numerica',
                        color_continuous_scale='RdYlGn'
                    )
                    fig_region_savings.update_layout(
                        yaxis_title="Economia Potencial (R$)",
                        showlegend=False
                    )
                    st.plotly_chart(fig_region_savings, use_container_width=True)
                    
            else:
                st.info("Nenhuma discrepância significativa encontrada com os filtros selecionados.")
    
    # Botão para download dos dados
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Exportar Dados")
    
    if st.sidebar.button("📊 Gerar Relatório"):
        # Preparar dados para export
        export_data = {
            'resumo_geral': {
                'total_registros': len(df_filtered),
                'produtos_unicos': df_filtered['codigo_br'].nunique(),
                'valor_total': df_filtered['valor_total'].sum(),
                'periodo': f"{df_filtered['compra'].min().strftime('%Y-%m-%d')} a {df_filtered['compra'].max().strftime('%Y-%m-%d')}"
            },
            'estatisticas_regionais': df_filtered.groupby('regiao').agg({
                'preco_unitario': ['count', 'mean', 'median'],
                'valor_total': 'sum'
            }).round(2).to_dict(),
            'top_discrepancias': calculate_price_discrepancies(df_filtered).head(20).to_dict('records')
        }
        
        # Converter para JSON e criar download
        import json
        report_json = json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
        
        st.sidebar.download_button(
            label="📥 Download Relatório JSON",
            data=report_json,
            file_name=f"relatorio_sus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
