# Reporter for GASKAP validation script
#

# Author James Dempsey
# Date 23 Nov 2019

from astropy.table import Table
from astropy.io.votable import from_table, writeto

class ReportSection(object):
    def __init__(self, title, target=None):
        self.title = title
        self.target = target
        self.items = []

    def add_item(self, title, value=None, link=None, image=None):
        item = ReportItem(title, value, link, image)
        self.items.append(item)


class ReportItem(object):
    def __init__(self, title, value=None, link=None, image=None):
        self.title = title
        self.value = value
        self.link = link
        self.image = image


class ValidationMetric(object):
    def __init__(self, title, description, value, status):
        self.title = title
        self.description = description
        self.value = value
        self.status = status


class ValidationReport(object):
    def __init__(self, title):
        self.title = title
        self.sections = []
        self.metrics = []

    def add_section(self, section):
        self.sections.append(section)

    def add_metric(self, metric):
        self.metrics.append(metric)

def _output_header(f, reporter):
    f.write('<html>\n<head>\n<title>{}</title>'.format(reporter.title))
    with open('style.html') as style:
        f.write(style.read())
    f.write('\n</head>\n<body>\n<h1 align="middle">{}</h1>'.format(reporter.title))
    return

def _output_report_table_header(f, title, target=None):
    f.write('\n<h2 align="middle">{}</h2>'.format(title))
    if target:
        f.write('\n<h4 align="middle"><i>File: \'{}\'</i></h4>'.format(target))
    f.write('\n<table class="reportTable">')
    return

def _get_metric_class_name(status):
    if status == 1:
        return 'good'
    elif status == 2:
        return 'uncertain'
    else:
        return 'bad'


def _output_metrics(f, reporter):
    _output_report_table_header(f, 'GASKAP HI Validation Metrics')
    f.write('\n<tr>')
    for metric in reporter.metrics:
        f.write('\n<th>{}</th>'.format(metric.title))
    f.write('\n</tr>\n<tr>')
    for metric in reporter.metrics:
        f.write('\n<td class={}>{}</td>'.format(_get_metric_class_name(metric.status), round(metric.value, 3)))
    f.write('\n</tr>\n<table>')
    return

def _output_section(f, section):
    _output_report_table_header(f, section.title, target=section.target)
    f.write('\n<tr>')
    for item in section.items:
        f.write('\n<th>{}</th>'.format(item.title))
    f.write('\n</tr>\n<tr>')
    for item in section.items:
        f.write('\n<td>')
        if item.link:
            f.write('<a href="{}">'.format(item.link))

        if item.value:
            f.write(str(item.value))
        elif item.image:
            f.write('<img src="{}">'.format(item.image))
        else:
            f.write('&nbsp;')

        if item.link:
            f.write('</a>')
        f.write('</td>')
    f.write('\n</tr>\n<table>')
    return

def _output_footer(f, reporter):
    f.write('\n\n</body>\n</html>')
    return


def output_html_report(reporter, dest_folder):
    with open(dest_folder + '/index.html', 'w') as f:
        _output_header(f, reporter)
        _output_metrics(f, reporter)
        for section in reporter.sections:
            _output_section(f, section)
        _output_footer(f, reporter)
    return


def output_metrics_xml(reporter, dest_folder):
    titles = []
    descs = []
    values = []
    statuses = []

    for metric in reporter.metrics:
        # title, description, value, status
        titles.append(metric.title)
        descs.append(metric.description)
        values.append(metric.value)
        statuses.append(metric.status)

    temp_table = Table([titles, descs, values, statuses], names=['metric_name', 'metric_description', 'metric_value', 'metric_status'])
    votable = from_table(temp_table)
    filename = dest_folder + '/gaskap-metrics.vot'
    writeto(votable, filename)
    return
