using LoveSense.Domaine;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace LoveSense.Service
{
    public interface ISessionVerificator
    {
        Task<IEnumerable<SessionVerification>> GetSessionsAsync();
    }
}
