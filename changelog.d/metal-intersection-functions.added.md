Metal output writes a ray intersection stage that reports hits with `ReportHit`
or `reportIntersectionEXT` as a Metal bounding-box intersection function that
returns the closest hit it reports inside the ray's interval, and Metal input
imports `[[intersection(bounding_box)]]` and `[[intersection(triangle)]]`
functions as CrossGL intersection stages. Both used to be refused.
