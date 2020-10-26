using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace LoveSense.Domaine
{
    public class Trace
    {
        private static TraceSource _source = new TraceSource("LoveSense.Domaine");

        public static void Journalise(string message)
        {
            _source.TraceEvent(TraceEventType.Information,0,message);
        }

    }
}
