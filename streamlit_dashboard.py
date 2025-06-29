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

def calculate_price_statistics(df_group):
    """Calculate comprehensive price statistics"""
    stats = {
        'preco_medio': df_group['preco_unitario'].mean(),
        'preco_mediano': df_group['preco_unitario'].median(),
        'preco_min': df_group['preco_unitario'].min(),
        'preco_max': df_group['preco_unitario'].max(),
        'desvio_padrao': df_group['preco_unitario'].std(),
        'coef_variacao': (df_group['preco_unitario'].std() / df_group['preco_unitario'].mean()) * 100 if df_group['preco_unitario'].mean() > 0 else 0,
        'percentil_25': df_group['preco_unitario'].quantile(0.25),
        'percentil_75': df_group['preco_unitario'].quantile(0.75),
        'gasto_total': df_group['preco_total'].sum(),
        'qtd_total': df_group['qtd_itens_comprados'].sum(),
        'num_compras': len(df_group),
        'num_fornecedores': df_group['fornecedor'].nunique()
    }
    return pd.Series(stats)

def calculate_savings_potential(df_filtered):
    """Calculate potential savings by using lowest prices in region"""
    savings_data = []
    
    for categoria in df_filtered['categoria'].unique():
        cat_data = df_filtered[df_filtered['categoria'] == categoria]
        
        if len(cat_data) < 2:
            continue
            
        # Group by region and calculate comprehensive stats
        region_stats = cat_data.groupby('regiao').apply(calculate_price_statistics).reset_index()
        
        for _, row in region_stats.iterrows():
            if row['preco_min'] < row['preco_medio'] and row['qtd_total'] > 0:
                # Savings using mean price
                economia_potencial_media = (row['preco_medio'] - row['preco_min']) * row['qtd_total']
                economia_percentual_media = ((row['preco_medio'] - row['preco_min']) / row['preco_medio']) * 100
                
                # Savings using median price  
                economia_potencial_mediana = (row['preco_mediano'] - row['preco_min']) * row['qtd_total']
                economia_percentual_mediana = ((row['preco_mediano'] - row['preco_min']) / row['preco_mediano']) * 100 if row['preco_mediano'] > 0 else 0
                
                savings_data.append({
                    'categoria': categoria,
                    'regiao': row['regiao'],
                    'preco_min': row['preco_min'],
                    'preco_medio': row['preco_medio'],
                    'preco_mediano': row['preco_mediano'],
                    'coef_variacao': row['coef_variacao'],
                    'gasto_total': row['gasto_total'],
                    'qtd_total': row['qtd_total'],
                    'economia_potencial_media': economia_potencial_media,
                    'economia_percentual_media': economia_percentual_media,
                    'economia_potencial_mediana': economia_potencial_mediana,
                    'economia_percentual_mediana': economia_percentual_mediana,
                    'num_fornecedores': row['num_fornecedores']
                })
    
    return pd.DataFrame(savings_data)

def create_brazil_choropleth(df_aggregated, value_column, title, color_scale='Viridis'):
    """Create choropleth map of Brazil by state"""
    fig = px.choropleth(
        df_aggregated,
        locations='uf',
        color=value_column,
        locationmode='geojson-id',
        scope='south america',
        color_continuous_scale=color_scale,
        title=title,
        hover_data=df_aggregated.columns.tolist()
    )
    
    # Focus on Brazil
    fig.update_geos(
        center=dict(lat=-14, lon=-55),
        projection_scale=4,
        showland=True,
        landcolor='lightgray'
    )
    
    return fig

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
        # Region filter with "Todos" option
        region_options = ["Todos"] + sorted(df_filtered['regiao'].unique().tolist())
        selected_region = st.selectbox(
            "Região",
            options=region_options,
            index=0
        )

        # Macro category filter with "Todos" option
        macro_cat_options = ["Todos"] + sorted(df_filtered['macro_categoria'].unique().tolist())
        selected_macro_category = st.selectbox(
            "Macro Categoria",
            options=macro_cat_options,
            index=0
        )

        # Filter dataframe based on selected macro category to get relevant submacro categories
        if selected_macro_category == "Todos":
            df_macro_filtered = df_filtered
        else:
            df_macro_filtered = df_filtered[df_filtered['macro_categoria'] == selected_macro_category]

        submacro_cat_options = ["Todos"] + sorted(df_macro_filtered['submacro_categoria'].unique().tolist())
        selected_submacro_category = st.selectbox(
            "Sub Categoria",
            options=submacro_cat_options,
            index=0
        )

        # Filter again based on selected submacro category to get relevant categories
        if selected_submacro_category == "Todos":
            df_submacro_filtered = df_macro_filtered
        else:
            df_submacro_filtered = df_macro_filtered[df_macro_filtered['submacro_categoria'] == selected_submacro_category]

        category_options = ["Todos"] + sorted(df_submacro_filtered['categoria'].unique().tolist())
        selected_category = st.selectbox(
            "Categoria Específica",
            options=category_options,
            index=0
        )

        # Apply filters only if not "Todos"
        if selected_region != "Todos":
            df_filtered = df_filtered[df_filtered['regiao'] == selected_region]
        
        if selected_macro_category != "Todos":
            df_filtered = df_filtered[df_filtered['macro_categoria'] == selected_macro_category]
        
        if selected_submacro_category != "Todos":
            df_filtered = df_filtered[df_filtered['submacro_categoria'] == selected_submacro_category]
        
        if selected_category != "Todos":
            df_filtered = df_filtered[df_filtered['categoria'] == selected_category]
    else:
        st.info("Filtros avançados desabilitados - mostrando todos os dados")

    st.markdown("---")
    st.markdown("### Filtro de Preços Extremos")
    
    # Enable/disable price filtering
    use_price_filter = st.checkbox("Filtrar preços extremos", value=False)
    
    if use_price_filter:
        # Calculate 99.9 percentile for current filtered data
        p999 = df_filtered['preco_unitario'].quantile(0.999)
        
        st.info(f"Percentil 99.9%: R$ {p999:.2f}")
        
        # Simple upper limit filter
        upper_limit = st.number_input(
            "Limite superior de preço (R$)",
            min_value=0.0,
            value=float(p999),
            step=0.01,
            format="%.2f",
            help="Preços acima deste valor serão removidos"
        )
        
        # Apply price filter
        before_filter_count = len(df_filtered)
        df_filtered = df_filtered[df_filtered['preco_unitario'] <= upper_limit]
        after_filter_count = len(df_filtered)
        
        # Show filtering impact
        removed_count = before_filter_count - after_filter_count
        removed_pct = (removed_count / before_filter_count * 100) if before_filter_count > 0 else 0
        
        if removed_count > 0:
            st.metric(
                "Registros removidos",
                f"{removed_count:,}".replace(',', '.'),
                delta=f"{removed_pct:.2f}% do total"
            )
        else:
            st.success("Nenhum registro removido com o limite atual")
    else:
        st.info("Filtro de preços desabilitado - todos os preços incluídos")

# Main dashboard
st.title("🏥 Dashboard SUS - Análise de Preços e Compras")
st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Visão Geral", 
    "📈 Variação de Preços", 
    "🗺️ Análise Regional", 
    "🌎 Visualização Geográfica",
    "💰 Economia Potencial"
])

with tab1:
    st.header("Visão Geral das Compras")
    
    # KPIs with mean vs median comparison
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
        median_price = df_filtered['preco_unitario'].median()
        delta_pct = ((avg_price - median_price) / median_price * 100) if median_price > 0 else 0
        st.metric("Preço Médio", format_currency(avg_price), delta=f"{delta_pct:.1f}% vs mediana")
    
    with col4:
        num_suppliers = df_filtered['fornecedor'].nunique()
        st.metric("Fornecedores", f"{num_suppliers:,}".replace(',', '.'))

    # Price distribution analysis
    st.subheader("📊 Análise de Distribuição de Preços")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Mean vs Median by category
        price_stats = df_filtered.groupby('submacro_categoria').agg({
            'preco_unitario': ['mean', 'median', 'std'],
            'preco_total': 'sum'
        }).round(2)
        price_stats.columns = ['preco_medio', 'preco_mediano', 'desvio_padrao', 'gasto_total']
        price_stats = price_stats.nlargest(15, 'gasto_total')
        
        # Calculate difference between mean and median
        price_stats['diferenca_media_mediana'] = price_stats['preco_medio'] - price_stats['preco_mediano']
        price_stats['percentual_diferenca'] = (price_stats['diferenca_media_mediana'] / price_stats['preco_mediano'] * 100)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Preço Médio',
            x=price_stats['preco_medio'],
            y=price_stats.index,
            orientation='h',
            marker_color='lightblue'
        ))
        fig.add_trace(go.Bar(
            name='Preço Mediano',
            x=price_stats['preco_mediano'],
            y=price_stats.index,
            orientation='h',
            marker_color='darkblue'
        ))
        fig.update_layout(
            title="Preço Médio vs Mediano por Categoria",
            height=500,
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top categories by spending
        top_categories = df_filtered.groupby('categoria')['preco_total'].sum().nlargest(10)
        fig = px.bar(
            x=top_categories.values,
            y=top_categories.index,
            orientation='h',
            title="Top 10 Categorias por Gasto Total"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Regional comparison
    st.subheader("🌍 Comparação Regional - Média vs Mediana")
    region_stats = df_filtered.groupby('regiao').apply(calculate_price_statistics).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            region_stats,
            x='regiao',
            y=['preco_medio', 'preco_mediano'],
            title="Preços Médio e Mediano por Região",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(
            region_stats,
            values='gasto_total',
            names='regiao',
            title="Distribuição do Gasto por Região"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Variação de Preços ao Longo do Tempo")
    
    # Select category for analysis
    category_options = ['Todos'] + list(df_filtered['macro_categoria'].unique())
    selected_cat = st.selectbox(
        "Selecione uma categoria para análise temporal:",
        options=category_options,
        index=0
    )
    
    # Filter data based on selection
    if selected_cat == 'Todos':
        cat_data = df_filtered
    else:
        cat_data = df_filtered[df_filtered['macro_categoria'] == selected_cat]
    
    cat_data = df_filtered[df_filtered['categoria'] == selected_cat]
    
    if not cat_data.empty:
        # Monthly price evolution with mean vs median
        monthly_stats = cat_data.groupby(cat_data['compra'].dt.to_period('M')).apply(calculate_price_statistics).reset_index()
        monthly_stats['compra'] = monthly_stats['compra'].dt.to_timestamp()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price evolution chart with confidence bands
            fig = go.Figure()
            
            # Add mean line
            fig.add_trace(go.Scatter(
                x=monthly_stats['compra'],
                y=monthly_stats['preco_medio'],
                mode='lines+markers',
                name='Preço Médio',
                line=dict(color='blue', width=3)
            ))
            
            # Add median line  
            fig.add_trace(go.Scatter(
                x=monthly_stats['compra'],
                y=monthly_stats['preco_mediano'],
                mode='lines+markers',
                name='Preço Mediano',
                line=dict(color='red', width=3)
            ))
            
            # Add quartile bands
            fig.add_trace(go.Scatter(
                x=monthly_stats['compra'],
                y=monthly_stats['percentil_75'],
                mode='lines',
                name='75° Percentil',
                line=dict(color='lightgray', dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=monthly_stats['compra'],
                y=monthly_stats['percentil_25'],
                mode='lines',
                name='25° Percentil',
                line=dict(color='lightgray', dash='dash'),
                fill='tonexty',
                fillcolor='rgba(200,200,200,0.2)'
            ))
            
            fig.update_layout(
                title=f"Evolução de Preços - {selected_cat}",
                xaxis_title="Período",
                yaxis_title="Preço (R$)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Coefficient of variation over time
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly_stats['compra'],
                y=monthly_stats['coef_variacao'],
                mode='lines+markers',
                name='Coeficiente de Variação (%)',
                line=dict(color='orange')
            ))
            fig.update_layout(
                title=f"Volatilidade de Preços - {selected_cat}",
                xaxis_title="Período",
                yaxis_title="Coeficiente de Variação (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Price volatility by supplier with enhanced metrics
        st.subheader("Análise Detalhada por Fornecedor")
        supplier_stats = cat_data.groupby('fornecedor').apply(calculate_price_statistics).reset_index()
        supplier_stats = supplier_stats[supplier_stats['num_compras'] >= 5].nlargest(15, 'gasto_total')
        
        if not supplier_stats.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(
                    supplier_stats,
                    x='preco_medio',
                    y='coef_variacao',
                    size='gasto_total',
                    color='num_fornecedores',
                    hover_data=['num_compras', 'preco_mediano'],
                    title="Preço Médio vs Volatilidade por Fornecedor"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Mean vs median scatter for suppliers
                fig = px.scatter(
                    supplier_stats,
                    x='preco_mediano',
                    y='preco_medio',
                    size='gasto_total',
                    hover_data=['fornecedor', 'num_compras'],
                    title="Preço Médio vs Mediano por Fornecedor"
                )
                # Add diagonal line for reference
                max_price = max(supplier_stats[['preco_medio', 'preco_mediano']].max())
                fig.add_shape(
                    type="line", line=dict(dash="dash", color="red"),
                    x0=0, x1=max_price, y0=0, y1=max_price
                )
                st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Análise Regional Detalhada")
    
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
        
        municipal_stats = uf_data.groupby('municipio_instituicao').apply(calculate_price_statistics).reset_index()
        municipal_stats = municipal_stats.nlargest(15, 'gasto_total').sort_values('gasto_total', ascending=True)
        
        fig = px.bar(
            municipal_stats,
            x='gasto_total',
            y='municipio_instituicao',
            orientation='h',
            title=f"Gasto por Município - {selected_uf}",
            hover_data=['qtd_total', 'num_fornecedores', 'preco_medio']
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Eficiência de Preços por Município")
        
        # Calculate price efficiency (lower mean-median difference = more efficient)
        municipal_stats['diferenca_media_mediana'] = municipal_stats['preco_medio'] - municipal_stats['preco_mediano']
        municipal_stats['eficiencia_preco'] = 1 / (1 + municipal_stats['diferenca_media_mediana'])
        
        top_efficient = municipal_stats.nlargest(10, 'eficiencia_preco')
        
        fig = px.bar(
            top_efficient,
            x='eficiencia_preco',
            y='municipio_instituicao',
            orientation='h',
            title="Municípios Mais Eficientes (Menor Diferença Média-Mediana)",
            color='coef_variacao',
            color_continuous_scale='RdYlBu_r'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    # Enhanced regional comparison
    st.subheader("Comparação Detalhada por Região")
    
    # Order categories by frequency (most frequent first)
    category_counts = df_filtered['submacro_categoria'].value_counts()
    ordered_categories = category_counts.index.tolist()
    
    comp_category = st.selectbox(
        "Categoria para comparação:",
        options=ordered_categories,
        key="regional_category"
    )
    
    regional_data = df_filtered[df_filtered['submacro_categoria'] == comp_category]
    
    if not regional_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(
                regional_data,
                x='regiao',
                y='preco_unitario',
                title=f"Distribuição de Preços por Região - {comp_category}",
                points="outliers"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Violin plot for better distribution visualization
            fig = px.violin(
                regional_data,
                x='regiao',
                y='preco_unitario',
                title=f"Densidade de Distribuição - {comp_category}",
                box=True
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced regional statistics table
        regional_detailed_stats = regional_data.groupby('regiao').apply(calculate_price_statistics).reset_index()
        
        # Format for display
        display_stats = regional_detailed_stats.copy()
        for col in ['preco_medio', 'preco_mediano', 'preco_min', 'preco_max', 'desvio_padrao', 'percentil_25', 'percentil_75']:
            display_stats[col] = display_stats[col].apply(lambda x: f"R$ {x:.2f}")
        display_stats['coef_variacao'] = display_stats['coef_variacao'].apply(lambda x: f"{x:.1f}%")
        
        st.subheader("Estatísticas Detalhadas por Região")
        st.dataframe(
            display_stats[['regiao', 'preco_medio', 'preco_mediano', 'preco_min', 'preco_max', 
                          'coef_variacao', 'num_compras', 'num_fornecedores']].rename(columns={
                'regiao': 'Região',
                'preco_medio': 'Preço Médio',
                'preco_mediano': 'Preço Mediano',
                'preco_min': 'Preço Mínimo',
                'preco_max': 'Preço Máximo',
                'coef_variacao': 'Coef. Variação',
                'num_compras': 'Nº Compras',
                'num_fornecedores': 'Nº Fornecedores'
            }),
            use_container_width=True
        )

with tab4:
    st.header("🌎 Visualização Geográfica")
    
    st.markdown("""
    Esta seção apresenta visualizações geográficas para identificar padrões regionais de preços, 
    gastos e oportunidades de economia em todo o território nacional.
    """)
    
    # Geographic analysis selector
    geo_analysis = st.selectbox(
        "Selecione o tipo de análise geográfica:",
        options=[
            "Gasto Total por Estado",
            "Preço Médio por Estado", 
            "Variabilidade de Preços (Coef. Variação)",
            "Eficiência de Compras",
            "Densidade de Fornecedores"
        ]
    )
    
    # Calculate state-level statistics
    state_stats = df_filtered.groupby('uf').apply(calculate_price_statistics).reset_index()
    state_stats['eficiencia_compras'] = state_stats['gasto_total'] / state_stats['preco_medio']
    state_stats['densidade_fornecedores'] = state_stats['num_fornecedores'] / state_stats['num_compras']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create choropleth map based on selected analysis
        if geo_analysis == "Gasto Total por Estado":
            fig = create_brazil_choropleth(
                state_stats, 'gasto_total', 
                'Gasto Total por Estado (R$)', 
                'Reds'
            )
        elif geo_analysis == "Preço Médio por Estado":
            fig = create_brazil_choropleth(
                state_stats, 'preco_medio',
                'Preço Médio por Estado (R$)',
                'Viridis'
            )
        elif geo_analysis == "Variabilidade de Preços (Coef. Variação)":
            fig = create_brazil_choropleth(
                state_stats, 'coef_variacao',
                'Coeficiente de Variação de Preços (%)',
                'RdYlBu_r'
            )
        elif geo_analysis == "Eficiência de Compras":
            fig = create_brazil_choropleth(
                state_stats, 'eficiencia_compras',
                'Eficiência de Compras (Volume/Preço)',
                'Blues'
            )
        else:  # Densidade de Fornecedores
            fig = create_brazil_choropleth(
                state_stats, 'densidade_fornecedores',
                'Densidade de Fornecedores (Fornecedores/Compra)',
                'Greens'
            )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top 10 Estados")
        
        if geo_analysis == "Gasto Total por Estado":
            top_states = state_stats.nlargest(10, 'gasto_total')[['uf', 'gasto_total']]
            top_states['gasto_total'] = top_states['gasto_total'].apply(format_currency)
        elif geo_analysis == "Preço Médio por Estado":
            top_states = state_stats.nlargest(10, 'preco_medio')[['uf', 'preco_medio']]
            top_states['preco_medio'] = top_states['preco_medio'].apply(format_currency)
        elif geo_analysis == "Variabilidade de Preços (Coef. Variação)":
            top_states = state_stats.nlargest(10, 'coef_variacao')[['uf', 'coef_variacao']]
            top_states['coef_variacao'] = top_states['coef_variacao'].apply(lambda x: f"{x:.1f}%")
        elif geo_analysis == "Eficiência de Compras":
            top_states = state_stats.nlargest(10, 'eficiencia_compras')[['uf', 'eficiencia_compras']]
            top_states['eficiencia_compras'] = top_states['eficiencia_compras'].apply(lambda x: f"{x:.2f}")
        else:
            top_states = state_stats.nlargest(10, 'densidade_fornecedores')[['uf', 'densidade_fornecedores']]
            top_states['densidade_fornecedores'] = top_states['densidade_fornecedores'].apply(lambda x: f"{x:.3f}")
        
        st.dataframe(top_states, use_container_width=True)
    
    # Geographic insights
    st.subheader("💡 Insights Geográficos")
    
    # Calculate some insights
    highest_price_state = state_stats.loc[state_stats['preco_medio'].idxmax()]
    lowest_price_state = state_stats.loc[state_stats['preco_medio'].idxmin()]
    highest_variation_state = state_stats.loc[state_stats['coef_variacao'].idxmax()]
    most_efficient_state = state_stats.loc[state_stats['eficiencia_compras'].idxmax()]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Estado com Maior Preço Médio", 
            highest_price_state['uf'],
            delta=format_currency(highest_price_state['preco_medio'])
        )
    
    with col2:
        st.metric(
            "Estado com Menor Preço Médio",
            lowest_price_state['uf'], 
            delta=format_currency(lowest_price_state['preco_medio'])
        )
    
    with col3:
        st.metric(
            "Estado com Maior Variação",
            highest_variation_state['uf'],
            delta=f"{highest_variation_state['coef_variacao']:.1f}%"
        )
    
    with col4:
        st.metric(
            "Estado Mais Eficiente",
            most_efficient_state['uf'],
            delta=f"{most_efficient_state['eficiencia_compras']:.2f}"
        )
    
    # Category-specific geographic analysis
    st.subheader("🗺️ Análise Geográfica por Categoria")
    
    selected_geo_category = st.selectbox(
        "Selecione uma categoria para análise geográfica:",
        options=df_filtered['submacro_categoria'].value_counts().head(20).index.tolist(),
        key="geo_category"
    )
    
    cat_geo_data = df_filtered[df_filtered['submacro_categoria'] == selected_geo_category]
    if not cat_geo_data.empty:
        cat_state_stats = cat_geo_data.groupby('uf').apply(calculate_price_statistics).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_brazil_choropleth(
                cat_state_stats, 'preco_medio',
                f'Preço Médio - {selected_geo_category}',
                'RdYlBu_r'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = create_brazil_choropleth(
                cat_state_stats, 'coef_variacao',
                f'Variabilidade - {selected_geo_category}',
                'Reds'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("Análise de Economia Potencial")
    
    st.markdown("""
    Esta análise identifica oportunidades de economia comparando preços médios e medianos dentro de cada região.
    Diferentes métricas oferecem perspectivas distintas sobre o potencial de economia.
    """)
    
    # Calculate enhanced savings
    savings_df = calculate_savings_potential(df_filtered)
    
    if not savings_df.empty:
        # Savings methodology selector
        savings_method = st.radio(
            "Método de Cálculo de Economia:",
            options=["Baseado na Média", "Baseado na Mediana"],
            help="Média: mais sensível a outliers. Mediana: mais robusta a valores extremos."
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top categories with highest savings potential
            if savings_method == "Baseado na Média":
                top_savings = savings_df.nlargest(15, 'economia_potencial_media')
                value_col = 'economia_potencial_media'
            else:
                top_savings = savings_df.nlargest(15, 'economia_potencial_mediana')
                value_col = 'economia_potencial_mediana'
            
            fig = px.bar(
                top_savings,
                x=value_col,
                y='categoria',
                color='regiao',
                orientation='h',
                title=f"Top 15 - Maior Potencial de Economia ({savings_method})"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Savings vs price variation analysis
            if savings_method == "Baseado na Média":
                x_col, y_col = 'economia_percentual_media', 'economia_potencial_media'
            else:
                x_col, y_col = 'economia_percentual_mediana', 'economia_potencial_mediana'
                
            fig = px.scatter(
                savings_df,
                x=x_col,
                y='coef_variacao',
                size=y_col,
                color='regiao',
                hover_data=['categoria', 'num_fornecedores'],
                title="% Economia vs Variabilidade de Preços"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Comparison between mean and median based savings
        st.subheader("📊 Comparação: Economia Baseada em Média vs Mediana")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                savings_df,
                x='economia_potencial_media',
                y='economia_potencial_mediana',
                color='regiao',
                size='gasto_total',
                hover_data=['categoria'],
                title="Economia Potencial: Média vs Mediana"
            )
            # Add diagonal reference line
            max_val = max(savings_df[['economia_potencial_media', 'economia_potencial_mediana']].max())
            fig.add_shape(
                type="line", line=dict(dash="dash", color="red"),
                x0=0, x1=max_val, y0=0, y1=max_val
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Calculate correlation between methods
            correlation = savings_df['economia_potencial_media'].corr(savings_df['economia_potencial_mediana'])
            
            fig = px.histogram(
                savings_df,
                x='economia_percentual_media',
                nbins=30,
                title="Distribuição do Percentual de Economia (Método da Média)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric(
                "Correlação entre Métodos",
                f"{correlation:.3f}",
                help="Correlação entre economia calculada pela média vs mediana"
            )
        
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
        
        if savings_method == "Baseado na Média":
            display_savings = display_savings.sort_values('economia_potencial_media', ascending=False)
        else:
            display_savings = display_savings.sort_values('economia_potencial_mediana', ascending=False)
        
        # Format the display table with both methods
        display_table = display_savings.copy()
        format_cols = ['preco_min', 'preco_medio', 'preco_mediano', 'gasto_total', 
                      'economia_potencial_media', 'economia_potencial_mediana']
        for col in format_cols:
            display_table[col] = display_table[col].apply(format_currency)
        
        display_table['economia_percentual_media'] = display_table['economia_percentual_media'].apply(lambda x: f"{x:.1f}%")
        display_table['economia_percentual_mediana'] = display_table['economia_percentual_mediana'].apply(lambda x: f"{x:.1f}%")
        display_table['coef_variacao'] = display_table['coef_variacao'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(
            display_table[['categoria', 'regiao', 'preco_min', 'preco_medio', 'preco_mediano',
                          'coef_variacao', 'gasto_total', 'economia_potencial_media', 
                          'economia_percentual_media', 'economia_potencial_mediana', 
                          'economia_percentual_mediana', 'num_fornecedores']].rename(columns={
                'categoria': 'Categoria',
                'regiao': 'Região', 
                'preco_min': 'Preço Mínimo',
                'preco_medio': 'Preço Médio',
                'preco_mediano': 'Preço Mediano',
                'coef_variacao': 'Coef. Variação',
                'gasto_total': 'Gasto Total',
                'economia_potencial_media': 'Economia (Média)',
                'economia_percentual_media': '% Economia (Média)',
                'economia_potencial_mediana': 'Economia (Mediana)', 
                'economia_percentual_mediana': '% Economia (Mediana)',
                'num_fornecedores': 'Nº Fornecedores'
            }),
            use_container_width=True
        )
        
        # Summary metrics with both methods
        st.markdown("### Resumo da Economia Potencial")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Método da Média")
            total_savings_mean = savings_df['economia_potencial_media'].sum()
            total_spending = savings_df['gasto_total'].sum()
            avg_savings_pct_mean = (total_savings_mean / total_spending * 100) if total_spending > 0 else 0
            
            subcol1, subcol2, subcol3 = st.columns(3)
            with subcol1:
                st.metric("Economia Total", format_currency(total_savings_mean))
            with subcol2:
                st.metric("Gasto Analisado", format_currency(total_spending))
            with subcol3:
                st.metric("% Economia Média", f"{avg_savings_pct_mean:.1f}%")
        
        with col2:
            st.markdown("#### Método da Mediana")
            total_savings_median = savings_df['economia_potencial_mediana'].sum()
            avg_savings_pct_median = (total_savings_median / total_spending * 100) if total_spending > 0 else 0
            
            subcol1, subcol2, subcol3 = st.columns(3)
            with subcol1:
                st.metric("Economia Total", format_currency(total_savings_median))
            with subcol2:
                st.metric("Diferença entre Métodos", format_currency(abs(total_savings_mean - total_savings_median)))
            with subcol3:
                st.metric("% Economia Média", f"{avg_savings_pct_median:.1f}%")
    
    else:
        st.warning("Não foi possível calcular economia potencial com os filtros atuais.")

# Footer
st.markdown("---")
st.markdown("Dashboard desenvolvido para análise de dados do Sistema Único de Saúde (SUS)")
