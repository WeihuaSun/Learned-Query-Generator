SELECT COUNT(*) FROM census WHERE workclass = 'Local-gov' AND education = '7th-8th' AND marital_status = 'Married-civ-spouse' AND occupation = 'Craft-repair' AND relationship = 'Unmarried' AND hours_per_week <= 68 AND native_country = 'United-States';