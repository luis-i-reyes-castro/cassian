#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 12:01:29 2018

@author: josemanuelvera
"""

from cassian.connectivity import DatabaseClient


client = DatabaseClient( store_id = 101)

client.fetch_data( intro_year_limit = 2016)