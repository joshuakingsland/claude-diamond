"""Resumable, per-game historical full-game odds capture (dry-run by default)."""
import argparse, csv, os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from config import MARKETS, PRICED_ODDS_REGIONS
from first_inning_odds import _append, _credit, _iso, _load_rows, events_on_day, events_url, response_events
from historical_odds import _request
from odds import QUOTE_FIELDS, _quote_rows, append_quote_log, paired_book_quotes
from first_inning_odds import EVENT_ODDS_API, _url

MANIFEST=Path('data/full_game_event_audit.csv'); QUOTES=Path('data/full_game_event_quotes.csv')
FIELDS=['audit_id','event_id','home_team','away_team','commence_time','snapshot_role','requested_snapshot','returned_snapshot','status','quote_count','odds_credits_used','discovery_credits_used','credits_remaining','error']
def days(start,end):
 d=date.fromisoformat(start); e=date.fromisoformat(end)
 while d<=e: yield d.isoformat(); d+=timedelta(days=1)
def url(key,event,snapshot,region): return _url(f'{EVENT_ODDS_API}/{event}/odds',key,regions=region,markets=','.join(MARKETS),oddsFormat='american',date=snapshot.strftime('%Y-%m-%dT%H:%M:%SZ'))
def run(key,start,end,max_events,lead_minutes=20,role='close',region='us',manifest=MANIFEST,quotes=QUOTES,dry_run=True):
 if not 1<=lead_minutes<=1440 or max_events<1: raise ValueError('invalid cap or lead')
 ds=list(days(start,end)); estimate=min(max_events,len(ds)*16)*30+len(ds)
 if dry_run: print(f'dry run: {start}..{end}, {role}, {lead_minutes}m lead; at most {max_events} event calls, ~{estimate} credits'); return []
 if not key: raise ValueError('ODDS_API_KEY is required')
 done={r['audit_id'] for r in _load_rows(manifest)}; selected=[]; discoveries=[]
 for day in ds:
  payload,h=_request(events_url(key,day)); discoveries.append(_credit(h.get('used')))
  for event in events_on_day(payload,day):
   snap=_iso(event['commence_time'])-timedelta(minutes=lead_minutes); aid='|'.join((event['id'],role,snap.strftime('%Y-%m-%dT%H:%M:%SZ'),region))
   if aid not in done and len(selected)<max_events: selected.append((event,snap,aid,_credit(h.get('used')))); h={'used':'0'}
  if len(selected)>=max_events: break
 rows=[]
 for i,(event,snap,aid,disc) in enumerate(selected):
  base={'audit_id':aid,'event_id':event['id'],'home_team':event.get('home_team',''),'away_team':event.get('away_team',''),'commence_time':event['commence_time'],'snapshot_role':role,'requested_snapshot':snap.strftime('%Y-%m-%dT%H:%M:%SZ'),'discovery_credits_used':disc}
  try:
   payload,h=_request(url(key,event['id'],snap,region)); priced=dict(event); got=response_events(payload)
   if got: priced.update(got[0])
   qs=paired_book_quotes(priced,region,MARKETS); append_quote_log(quotes,_quote_rows(priced,qs,payload.get('timestamp') or base['requested_snapshot']))
   row={**base,'returned_snapshot':payload.get('timestamp',''),'status':'offered' if qs else 'no_offer','quote_count':len(qs),'odds_credits_used':_credit(h.get('used')),'credits_remaining':h.get('remaining',''),'error':''}
  except Exception as err: row={**base,'returned_snapshot':'','status':'failed','quote_count':0,'odds_credits_used':0,'credits_remaining':'','error':repr(err)}
  _append(manifest,FIELDS,[row]); rows.append(row)
 return rows
def main():
 p=argparse.ArgumentParser(); p.add_argument('--start',required=True);p.add_argument('--end',required=True);p.add_argument('--max-events',type=int,default=100);p.add_argument('--lead-minutes',type=int,default=20);p.add_argument('--snapshot-role',choices=['early','close'],default='close');p.add_argument('--dry-run',action='store_true');a=p.parse_args();run(os.getenv('ODDS_API_KEY'),a.start,a.end,a.max_events,a.lead_minutes,a.snapshot_role,dry_run=a.dry_run)
if __name__=='__main__': main()
