import 'server-only';
import { cookies } from 'next/headers';

const COOKIE = 'fr_uid';

/**
 * Anonymous per-browser id used to namespace uploads until real auth is added.
 * Replace with your auth provider's user id (see docs/ARCHITECTURE.md, "Auth").
 */
export async function userId(): Promise<string> {
  const jar = await cookies();
  let id = jar.get(COOKIE)?.value;
  if (!id || !/^[a-z0-9]{16,32}$/.test(id)) {
    id = crypto.randomUUID().replace(/-/g, '');
    jar.set(COOKIE, id, { httpOnly: true, sameSite: 'lax', secure: process.env.NODE_ENV === 'production', maxAge: 31_536_000, path: '/' });
  }
  return id;
}
