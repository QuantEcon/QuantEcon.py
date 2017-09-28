"""
Filename: timing.py
Authors: Pablo Winant
Date: 10/16/14
Provides Matlab-like tic, tac and toc functions.
"""


class __Timer__:
    """Computes elapsed time, between tic, tac, and toc.

    Methods
    -------
    tic :
        Resets timer.
    toc :
        Returns and prints time elapsed since last tic().
    tac :
        Returns and prints time elapsed since last
             tic(), tac() or toc() whichever occured last.
    loop_timer :
        Returns and prints the total and average time elapsed for n runs 
        of a given function.
    """

    start = None
    last = None

    def tic(self):
        """Resets timer."""

        import time

        t = time.time()
        self.start = t
        self.last = t

    def tac(self, digits=2, print_output=True, output=True):
        """
        Returns and prints time elapsed since last tic(), tac() or toc()
        whichever occured last.
        
        Parameters
        ----------
        digits : scalar(int), optional(default=2)
            Number of digits for time elapsed.
        
        print_output : bool, optional(default=True)
            If True, then prints time.
        
        output : bool, optional(default=True)
            If True, then returns time.

        Returns
        -------
        tac : str
            Time elapsed since last tic(), tac() or toc().
        
        """
        
        import time

        if self.start is None:
            raise Exception("tac() without tic()")

        t = time.time()
        elapsed = t-self.last
        self.last = t
        
        if print_output:
            m, s = divmod(elapsed, 60)
            h, m = divmod(m, 60)
            print("TAC: Elapsed: %d:%02d:%0d.%0*d" % 
                  (h, m, s, digits, (s%1)*(10**digits)))
        
        if output:
            rounded_time = str(round(elapsed, digits))
            time_digits = rounded_time.split('.')[1]

            while len(time_digits) < digits:
                time_digits += "0"
            
            tac = rounded_time.split('.')[0] + "." + time_digits
            
            return tac

    def toc(self, digits=2, print_output=True, output=True):
        """
        Returns and prints time elapsed since last tic().
        
        Parameters
        ----------
        digits : scalar(int), optional(default=2)
            Number of digits for time elapsed.
        
        print_output : bool, optional(default=True)
            If True, then prints time.
        
        output : bool, optional(default=True)
            If True, then returns time.
        
        Returns
        -------
        toc : str
            Time elapsed since last tic().
        
        """

        import time

        if self.start is None:
            raise Exception("toc() without tic()")

        t = time.time()
        self.last = t
        elapsed = t-self.start

        if print_output:
            m, s = divmod(elapsed, 60)
            h, m = divmod(m, 60)
            print("TOC: Elapsed: %d:%02d:%0d.%0*d" %
                  (h, m, s, digits, (s%1)*(10**digits)))
        
        if output:
            rounded_time = str(round(elapsed, digits))
            time_digits = rounded_time.split('.')[1]

            while len(time_digits) < digits:
                time_digits += "0"

            toc = rounded_time.split('.')[0] + "." + time_digits
                
            return toc
        

    def loop_timer(self, n, function, args=None, digits=2, print_output=True,
                   output=True, best_of=3):
        """
        Returns and prints the total and average time elapsed for n runs 
        of function.
                
        Parameters
        ----------
        n : scalar(int)
            Number of runs.
        
        function : function
            Function to be timed.
        
        args : list, optional(default=None)
            Arguments of the function.
        
        digits : scalar(int), optional(default=2)
            Number of digits for time elapsed.
            
        print_output : bool, optional(default=True)
            If True, then prints average time.
        
        output : bool, optional(default=True)
            If True, then returns average time.
        
        best_of : scalar(int), optional(default=3)
            Average time over best_of runs.

        Returns
        -------
        average_time : str
            Average time elapsed for n runs of function.
        
        average_of_best : str
            Average of best_of times for n runs of function.
            
        """
        tic()
        all_times = []
        for run in range(n):
            if hasattr(args, '__iter__'):
                function(*args)
            elif args == None:
                function()
            else:
                function(args)
            all_times.append(float(tac(digits, False, True)))

        elapsed = toc(digits, False, True)

        m, s = divmod(float(elapsed), 60)
        h, m = divmod(m, 60)

        print("Total run time: %d:%02d:%0d.%0*d" % 
              (h, m, s, digits, (s%1)*(10**digits)))

        average_time = sum(all_times) / len(all_times)
        
        best_times = all_times[:best_of]
        average_of_best = sum(best_times) / len(best_times)
        
        if print_output:
            m, s = divmod(average_time, 60)
            h, m = divmod(m, 60)
            print("Average time for %d runs: %d:%02d:%0d.%0*d" %
                               (n, h, m, s, digits, (s%1)*(10**digits)))
            m, s = divmod(average_of_best, 60)
            h, m = divmod(m, 60)
            print("Average of %d best times: %d:%02d:%0d.%0*d" %
                               (best_of, h, m, s, digits, (s%1)*(10**digits)))
        
        if output:
            rounded_time = str(round(average_time, digits))
            time_digits = rounded_time.split('.')[1]

            while len(time_digits) < digits:
                time_digits += "0"

            average_time = rounded_time.split('.')[0] + "." + time_digits
            
            rounded_time = str(round(average_of_best, digits))
            time_digits = rounded_time.split('.')[1]

            while len(time_digits) < digits:
                time_digits += "0"

            average_of_best = rounded_time.split('.')[0] + "." + time_digits
            
            return average_time, average_of_best

        
__timer__ = __Timer__()


def tic():
    """Saves time for future use with tac or toc."""
    return __timer__.tic()


def tac(digits=2, print_output=True, output=True):
    """Prints and returns elapsed time since last tic, tac or toc."""
    return __timer__.tac(digits, print_output, output)


def toc(digits=2, print_output=True, output=True):
    """Returns and prints time elapsed since last tic()."""
    return __timer__.toc(digits, print_output, output)


def loop_timer(n, function, args=None, digits=2, print_output=True,
               output=True, best_of=3):
    """Prints the total and average time elapsed for n runs of function."""
    return __timer__.loop_timer(n, function, args, digits, print_output, output,
                                best_of)