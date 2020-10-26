using System;

namespace LoveSense.Presentation.Web.Models
{
    public partial class SessionVerificationModel
    {
        public DateTime DateSession { get; set; }
        public bool Verdict { get; set; }
        public string Text { get; set; }
        public decimal Score { get; set; }
    }
}
