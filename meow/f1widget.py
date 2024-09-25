import fastf1
import pandas as pd

events = fastf1.get_events_remaining()
events = pd.DataFrame(events)
print(events)
events.to_csv('futureEvents.csv', index=False)

#load track data for next race
x= events["OfficialEventName"]
event = x.iloc[0]

track = fastf1.get_event(2024, event)
print(track)

#load track data for next race
# x= events["OfficialEventName"]
# event = ''
# for _ in x:
#     event = _
#     break
# track = fastf1.get_event(2024, event)
