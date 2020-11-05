import csv


class CSVLogger():
    def __init__(self, args, fieldnames, filename='log.csv'):
        """
        Initialize a csv file.

        Args:
            self: (todo): write your description
            fieldnames: (str): write your description
            filename: (str): write your description
        """

        self.filename = filename
        self.csv_file = open(filename, 'w')

        # Write model configuration at top of csv
        writer = csv.writer(self.csv_file)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])
        writer.writerow([''])

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        """
        Writes a row.

        Args:
            self: (todo): write your description
            row: (todo): write your description
        """
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        """
        Close the underlying file.

        Args:
            self: (todo): write your description
        """
        self.csv_file.close()
