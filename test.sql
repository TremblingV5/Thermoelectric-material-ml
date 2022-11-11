SELECT count(*)
FROM attendence
WHERE date > {} and date < {}
GROUP BY date;