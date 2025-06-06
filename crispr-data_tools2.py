#!/usr/bin/env python3

import os
import pandas as pd

INDEX_FILE = "/home/ubuntu/data/biogrid/BIOGRID-ORCS-SCREEN_INDEX-1.1.17.index.tab.txt"
SCREEN_DIR = "/home/ubuntu/data/biogrid"

class BioGRIDIndex:
    def __init__(self, index_path=INDEX_FILE):
        self.df = pd.read_csv(index_path, sep='\t', comment='#', low_memory=False)
        self.df.columns = [c.strip('#') for c in self.df.columns]

    def get_screen_metadata(self, screen_id):
        row = self.df[self.df['SCREEN_ID'] == int(screen_id)]
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    def filter_screens(self, **kwargs):
        query = self.df
        for k, v in kwargs.items():
            query = query[query[k] == v]
        return query['SCREEN_ID'].tolist()

    def get_significance_criteria(self, screen_id):
        row = self.df[self.df['SCREEN_ID'] == int(screen_id)]
        if row.empty:
            return None
        row = row.iloc[0]
        score_types = [row[f'SCORE.{i}_TYPE'] for i in range(1, 6) if f'SCORE.{i}_TYPE' in row and pd.notnull(row[f'SCORE.{i}_TYPE'])]
        return {
            "indicator": row['SIGNIFICANCE_INDICATOR'],
            "criteria": row['SIGNIFICANCE_CRITERIA'],
            "score_types": score_types
        }

class BioGRIDScreen:
    def __init__(self, screen_path):
        self.df = pd.read_csv(screen_path, sep='\t', comment='#', low_memory=False)

    def get_gene_row(self, gene_symbol):
        return self.df[self.df['OFFICIAL_SYMBOL'] == gene_symbol]

    def get_hits(self):
        return self.df[self.df['HIT'] == 'YES']

    def get_all(self):
        return self.df

def get_screen_path(screen_id, screen_dir=SCREEN_DIR):
    return os.path.join(screen_dir, f"BIOGRID-ORCS-SCREEN_{screen_id}-1.1.17.screen.tab.txt")

def query_gene_across_screens(gene_symbol, index: BioGRIDIndex, screen_dir=SCREEN_DIR):
    results = []
    for screen_id in index.df['SCREEN_ID']:
        screen_path = get_screen_path(screen_id, screen_dir)
        if not os.path.exists(screen_path):
            continue
        screen = BioGRIDScreen(screen_path)
        gene_row = screen.get_gene_row(gene_symbol)
        if not gene_row.empty:
            meta = index.get_screen_metadata(screen_id)
            results.append({
                "screen_id": screen_id,
                "metadata": meta,
                "gene_result": gene_row.to_dict('records')
            })
    return results

def query_screens_by_metadata(index: BioGRIDIndex, screen_dir=SCREEN_DIR, **filters):
    screen_ids = index.filter_screens(**filters)
    summaries = []
    for screen_id in screen_ids:
        screen_path = get_screen_path(screen_id, screen_dir)
        if not os.path.exists(screen_path):
            continue
        screen = BioGRIDScreen(screen_path)
        hits = screen.get_hits()
        meta = index.get_screen_metadata(screen_id)
        summaries.append({
            "screen_id": screen_id,
            "metadata": meta,
            "num_hits": len(hits),
            "top_hits": hits.sort_values(by='SCORE.1', ascending=False).head(10).to_dict('records')
        })
    return summaries

def summarize_screen(screen_id, index: BioGRIDIndex, screen_dir=SCREEN_DIR):
    screen_path = get_screen_path(screen_id, screen_dir)
    if not os.path.exists(screen_path):
        return None
    screen = BioGRIDScreen(screen_path)
    hits = screen.get_hits()
    meta = index.get_screen_metadata(screen_id)
    criteria = index.get_significance_criteria(screen_id)
    return {
        "screen_id": screen_id,
        "metadata": meta,
        "criteria": criteria,
        "num_hits": len(hits),
        "top_hits": hits.sort_values(by='SCORE.1', ascending=False).head(10).to_dict('records')
    }

def explain_significance(screen_id, index: BioGRIDIndex):
    criteria = index.get_significance_criteria(screen_id)
    if not criteria:
        return "No criteria found."
    return f"Hits are defined by: {criteria['indicator']} with criteria: {criteria['criteria']}. Score columns: {criteria['score_types']}"