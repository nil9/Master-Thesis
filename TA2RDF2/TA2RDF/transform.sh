
FILES=`find . -type f -name '*.xml'`

for f in $FILES
do
	wine msxsl.exe "$f" convertTransaction2RDFTriples.xsl -o "${f%.*}.txt"
	rm "${f%.*}.xml"
done





