Metal input translates corpus sources faster: the importer finds a source's
concrete struct definitions once per source text, and sanitizes each template
name once, instead of repeating both scans for every query. The output is
byte-identical; on 154 corpus units the import takes 22 % less CPU time (#4846).
