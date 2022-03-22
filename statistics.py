from obspy import UTCDateTime
import re

def sum_count_Mo(cata,starttime,endtime):
    """
    Caculte cumulated earthquake numbers and Moment
    """
    job = {}
    looptime = starttime
    day = 1
    while looptime < endtime:
        year = looptime.year
        if year not in job.keys():
            job[year] = {}
        _date = looptime.strftime("%Y%m%d")
        job[year][_date] = {}
        job[year][_date]['day'] = day
        job[year][_date]['count'] = 0
        job[year][_date]['Mo'] = 0
        looptime += 24*60*60
        day += 1
    for evid in cata.keys:
        etime = cata[evid][4]
        year = etime.year
        _edate = etime.strftime("%Y%m%d")
        Mw = cata[evid][3]
        Mo=pow(10,1.5*Mw+9.1)
        job[year][_edate]['count']+=1
        job[year][_edate]['Mo']+=Mo
    return job

def write_sum_count_Mo(job,outfile,mode="day"):
    """
    Write out cumulated results
    columns: float_year|date|day|day/year_count|cumulative_count|day_year_Mo|cumulative_Mo
    mode: "year","month" or "day"
    """
    if mode not in ["day","month","year"]:
        raise Exception("Mode should be 'day','month','year'")
    count_sum=0
    Mo_sum=0
    f=open(outfile,'w')
    for year in job.keys():
        pyear = UTCDateTime(year,1,1)   # present year head
        nyear = UTCDateTime(year+1,1,1) # next year head
        year_secs = nyear - pyear
        year_count = 0
        year_Mo = 0
        month_count = 0
        month_Mo = 0
        for _date in job[year].keys():
            datetime = UTCDateTime.strptime(_date,'%Y%m%d')
            float_year = year + (datetime-pyear)/year_secs
            day = job[year][_date]['day']
            count = job[year][_date]['count']
            year_count+=count
            month_count+=count
            count_sum += count
            Mo = job[year][_date]['Mo']
            year_Mo += Mo
            month_Mo += Mo
            Mo_sum += Mo
            #----------------------month mode---------------------
            if mode=="month":
                next_datetime = datetime+24*60*60
                _next_date = next_datetime.strftime('%Y%m%d')
                if _next_date[6:8] == "01": # means this day is the end of month
                    f.write(f"{format(float_year,'8.3f')} ")
                    f.write(f"{_date} {day} {month_count} {count_sum} ")
                    f.write(f"{int(month_Mo)} {int(Mo_sum)}\n")
                month_Mo = 0       # re-initiate
                month_count = 0
            #----------------------day mode-----------------------
            if mode=="day":
                f.write(f"{format(float_year,'8.3f')} ")
                f.write(f"{_date} {day} {count} {count_sum} ")
                f.write(f"{int(Mo)} {int(Mo_sum)}\n")
        if mode=="year":
            f.write(f"{format(float_year,'8.3f')} ")
            f.write(f"{_date} {day} {year_count} {count_sum} ")
            f.write(f"{int(year_Mo)} {int(Mo_sum)}\n")
    f.close()
    print("Output file is: ",outfile)
