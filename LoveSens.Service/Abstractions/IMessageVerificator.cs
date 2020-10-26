using LoveSense.Domaine;
using System.Threading.Tasks;

namespace LoveSense.Service
{
    public interface IMessageVerificator
    {
        Task<ResponseVerify> VerifyAsync(string text);
    }
}
