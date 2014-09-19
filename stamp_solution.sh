archive_name="$1.tar.gz"
time=`date +"%d.%m_%k_%M"`
echo $archive_name 'created'
tar -cf $archive_name solution.py description.txt
mv ./$archive_name ./solutions/$archive_name
echo "$1_$time" >> ./solutions/stats.txt
