import pandas as pd
import duckdb
from rapidfuzz import process, fuzz
import streamlit as st

# --- Configuration ---
S3_BUCKET = "bioinf-trailbiomed"
S3_PREFIX = "data/dev-crispr/TEST-SCREEN/SCREENS/"
INDEX_KEY = "data/dev-crispr/TEST-SCREEN/index.txt"
REGION = "us-west-2"

# --- DuckDB S3 Connection ---

@st.cache_resource
def duckdb_aws_setup(region=REGION):
    con = duckdb.connect()
    con.execute("INSTALL aws;")
    return con

def get_duckdb_s3_connection(region=REGION):
    # Use the cached connection for setup, but return a new connection for each query
    duckdb_aws_setup(region)
    con = duckdb.connect()
    con.execute("LOAD aws;")
    con.execute(f"SET s3_region = '{region}';")
    con.execute("""
        CREATE SECRET (
            TYPE S3,
            PROVIDER credential_chain
        );
    """)
    return con

# --- Load Index Metadata ---
def load_index_df(index_s3_path=f's3://{S3_BUCKET}/{INDEX_KEY}', region=REGION):
    with get_duckdb_s3_connection(region=region) as con:
        index_df = con.execute(f"SELECT * FROM read_csv_auto('{index_s3_path}', header=True, sep='\t')").df()
    return index_df

index_df = load_index_df()

# --- Controlled Vocabularies ---
virus_options = index_df['CONDITION_NAME'].dropna().unique().tolist() if 'CONDITION_NAME' in index_df.columns else []
cell_line_options = index_df['CELL_LINE'].dropna().unique().tolist() if 'CELL_LINE' in index_df.columns else []
species_options = index_df['ORGANISM_OFFICIAL'].dropna().unique().tolist() if 'ORGANISM_OFFICIAL' in index_df.columns else []

# --- Query Functions ---
def load_screen_tab_filtered(
    screen_id,
    bucket=S3_BUCKET,
    prefix=S3_PREFIX,
    gene_symbols=None,
    hit_value=None,
    score2_max=None
):
    s3_path = f's3://{bucket}/{prefix}TEST-SCREEN_{screen_id}-0.0.0.screen.tab.txt'
    where_clauses = []
    if gene_symbols is not None:
        if isinstance(gene_symbols, str):
            gene_symbols = [gene_symbols]
        gene_list = ','.join([f"'{g}'" for g in gene_symbols])
        where_clauses.append(f"OFFICIAL_SYMBOL IN ({gene_list})")
    if hit_value is not None:
        where_clauses.append(f"HIT = '{hit_value}'")
    if score2_max is not None:
        where_clauses.append(f'CAST("SCORE.2 (pos_fdr)" AS DOUBLE) <= {score2_max}')
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    query = f"SELECT *, {screen_id} AS SCREEN_ID FROM read_csv_auto('{s3_path}', header=True, sep='\t') {where_sql}"
    with get_duckdb_s3_connection() as con:
        return con.execute(query).df()

def get_results_by_genes(gene_symbols, index_df=index_df, bucket=S3_BUCKET, prefix=S3_PREFIX):
    if isinstance(gene_symbols, str):
        gene_symbols = [gene_symbols]
    results = []
    for screen_id in index_df['SCREEN_ID'].unique():
        df = load_screen_tab_filtered(screen_id, bucket, prefix, gene_symbols=gene_symbols)
        if not df.empty:
            meta = index_df[index_df['SCREEN_ID'] == screen_id]
            for col in meta.columns:
                if col not in df.columns:
                    df.loc[:, col] = meta.iloc[0][col]
            results.append(df)
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()

def get_results_by_screens(screen_ids, bucket=S3_BUCKET, prefix=S3_PREFIX):
    if isinstance(screen_ids, (str, int)):
        screen_ids = [screen_ids]
    results = []
    for screen_id in screen_ids:
        df = load_screen_tab_filtered(screen_id, bucket, prefix)
        if not df.empty:
            results.append(df)
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()

def get_results_by_condition(virus_query, index_df=index_df, bucket=S3_BUCKET, prefix=S3_PREFIX, virus_col='CONDITION_NAME', threshold=70):
    exact = index_df[index_df[virus_col].str.lower() == virus_query.lower()]
    if not exact.empty:
        screen_ids = exact['SCREEN_ID'].unique()
    else:
        choices = index_df[virus_col].dropna().unique()
        best_match, score, _ = process.extractOne(virus_query, choices, scorer=fuzz.token_sort_ratio)
        if score > threshold:
            screen_ids = index_df[index_df[virus_col] == best_match]['SCREEN_ID'].unique()
        else:
            return pd.DataFrame()  # No match found
    return get_results_by_screens(screen_ids, bucket, prefix)

def list_viruses():
    return virus_options

def list_cell_lines():
    return cell_line_options

def list_species():
    return species_options
