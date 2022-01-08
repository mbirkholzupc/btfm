function add_header(filename, header)

outfile = fopen('tmpfile', 'w');
fprintf(outfile, header);
fclose(outfile);

addstring = strcat(['cat tmpfile ', filename, ' > tmpfile2']);
system(addstring);
mvstring = strcat(['mv tmpfile2 ', filename]);
system(mvstring);
system('rm tmpfile');

