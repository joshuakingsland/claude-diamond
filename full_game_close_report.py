"""Research-only integrity report for per-game full-game closing snapshots."""
import argparse, json
from pathlib import Path
import pandas as pd

# Verified historical-event coverage from the provider, not a theoretical
# MLB schedule count.  The latter includes games this endpoint cannot return.
TARGET = 10354
def build(audit, quotes):
    if audit.empty: return {"status":"waiting_for_data","events":0}
    a=audit.copy(); a["lead_minutes"]=(pd.to_datetime(a["commence_time"],utc=True)-pd.to_datetime(a["requested_snapshot"],utc=True)).dt.total_seconds()/60
    offered=a[a.status.eq("offered")]
    result={"status":"research_only","target_events":TARGET,"attempted_events":int(len(a)),"offered_events":int(len(offered)),"failed_events":int(a.status.eq("failed").sum()),"no_offer_events":int(a.status.eq("no_offer").sum()),"completion_rate":round(len(a)/TARGET,5),"offer_rate":round(len(offered)/max(1,len(a)),5),"lead_minutes_median":round(float(offered.lead_minutes.median()),3),"pregame_requested_rate":round(float((offered.lead_minutes>0).mean()),5),"note":"Integrity and coverage only. No model selection, bets, or performance claims."}
    if not quotes.empty and {"event_id","market","book_key"}.issubset(quotes):
        q=quotes.drop_duplicates(["event_id","market","book_key"]); result["paired_quotes"]=int(len(quotes)); result["mean_books_per_market"] = round(float(q.groupby(["event_id","market"]).book_key.nunique().mean()),3)
    return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--audit',default='data/full_game_event_audit.csv');p.add_argument('--quotes',default='data/full_game_event_quotes.csv');p.add_argument('--out',default='full_game_close_evidence.json');a=p.parse_args()
 load=lambda x: pd.read_csv(x) if Path(x).exists() else pd.DataFrame()
 r=build(load(a.audit),load(a.quotes));Path(a.out).write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':main()
